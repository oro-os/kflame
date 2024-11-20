use std::{
	collections::HashMap,
	fs::File,
	io::{BufWriter, Write},
	sync::{
		Arc, Mutex,
		atomic::{AtomicBool, AtomicUsize, Ordering::SeqCst},
	},
};

use anyhow::Result;
use ctor::ctor;
use elf::{ElfBytes, endian::AnyEndian};
use nonoverlapping_interval_tree::NonOverlappingIntervalTree;
use qemu_plugin::{
	CallbackFlags, PluginId, TranslationBlock, VCPUIndex,
	install::{Args, Info, Value},
	plugin::{HasCallbacks, PLUGIN, Plugin, Register},
};

#[derive(Debug, Clone, Eq)]
struct Symbol {
	base: u64,
	name: String,
}

impl PartialEq for Symbol {
	fn eq(&self, other: &Self) -> bool {
		self.base == other.base
	}
}

struct Vcpu {
	icount: AtomicUsize,
	flame:  Arc<Mutex<FlameWriter>>,
}

struct FlameWriter {
	file: BufWriter<File>,
	stack: Vec<Symbol>,
	last_commit_count: usize,
}

#[derive(Default)]
struct Kflame {
	elf_path: String,
	elf_entry_addr: u64,
	buffer_capacity: usize,
	has_hit_entry_point: Arc<AtomicBool>,
	symtab: NonOverlappingIntervalTree<u64, Symbol>,
	vcpus: Arc<HashMap<VCPUIndex, Vcpu>>,
}

impl FlameWriter {
	pub fn new(elf_file: &str, vcpu_id: VCPUIndex, buffer_capacity: usize) -> Result<Self> {
		assert_ne!(buffer_capacity, 0);

		let filepath = format!("{elf_file}.flame-{vcpu_id}.txt");
		let file = BufWriter::with_capacity(buffer_capacity, File::create(filepath)?);

		Ok(Self {
			file,
			stack: Vec::new(),
			last_commit_count: 0,
		})
	}

	pub fn hit(&mut self, sym: &Option<Symbol>, vaddr: u64, icount: usize) {
		if let Some(sym) = sym {
			if sym.base == vaddr {
				self.commit_or_panic(icount);
				self.stack.push(sym.clone());
			} else {
				while let Some(last) = self.stack.last() {
					if sym == last {
						break;
					} else {
						self.commit_or_panic(icount);
						self.stack.pop();
					}
				}
			}
		} else {
			self.commit_or_panic(icount);
			self.reset(icount);
		}
	}

	pub fn reset(&mut self, icount: usize) {
		self.stack.clear();
		self.last_commit_count = icount;
	}

	fn commit_or_panic(&mut self, icount: usize) {
		self.commit(icount).expect("failed to commit flamegraph");
	}

	fn commit(&mut self, icount: usize) -> Result<()> {
		if icount == self.last_commit_count || self.stack.is_empty() {
			return Ok(());
		}

		for (i, sym) in self.stack.iter().enumerate() {
			if i > 0 {
				self.file.write(b";")?;
			}

			self.file.write(sym.name.as_bytes())?;
		}

		writeln!(self.file, " {}", icount - self.last_commit_count)?;
		self.last_commit_count = icount;

		Ok(())
	}
}

impl Register for Kflame {
	fn register(&mut self, _id: PluginId, args: &Args, _info: &Info) -> Result<()> {
		self.buffer_capacity =
			if let Some(Value::Integer(buffer_capacity)) = args.parsed.get("buffer") {
				*buffer_capacity as usize
			} else {
				1024 * 16
			};

		println!("kflame: buffer capacity set to {}", self.buffer_capacity);

		let Some(Value::String(elf_path)) = args.parsed.get("elf") else {
			return Err(anyhow::anyhow!(
				"plugin argument 'elf' not provided or is not string"
			));
		};

		self.elf_path = elf_path.clone();

		let elf_contents = std::fs::read(elf_path)
			.map_err(|e| anyhow::anyhow!("failed to read elf file at '{elf_path}': {e}"))?;

		let elf = ElfBytes::<AnyEndian>::minimal_parse(&elf_contents)
			.map_err(|e| anyhow::anyhow!("failed to parse elf file at '{elf_path}': {e}"))?;

		self.elf_entry_addr = elf.ehdr.e_entry;

		let Some((symtab, strtab)) = elf.symbol_table().map_err(|e| {
			anyhow::anyhow!("failed to get symbol table from elf file at '{elf_path}': {e}")
		})?
		else {
			return Err(anyhow::anyhow!(
				"elf file at '{elf_path}' has no symbol table"
			));
		};

		for sym in symtab {
			let st_name = strtab.get(sym.st_name as usize).map_err(|e| {
				anyhow::anyhow!(
					"failed to get symbol name at offset {} from elf file at '{elf_path}': {e}",
					sym.st_name
				)
			})?;
			let st_value = sym.st_value;
			let st_size = sym.st_size;

			let demangled = format!("{:#}", rustc_demangle::demangle(&st_name));

			if st_value == 0 || st_size == 0 {
				continue;
			}

			let existing = self.symtab.insert_replace(
				st_value..(st_value + st_size),
				Symbol {
					base: st_value,
					name: demangled,
				},
			);

			if existing.len() > 0 {
				return Err(anyhow::anyhow!(
					"symbol at {:#x}..{:#x} overlaps with existing symbol(s): {:#?}",
					st_value,
					st_value + st_size,
					existing
				));
			}
		}

		println!(
			"kflame: registered {} symbols from '{}'",
			self.symtab.len(),
			elf_path
		);

		Ok(())
	}
}

impl HasCallbacks for Kflame {
	fn on_vcpu_init(&mut self, _id: PluginId, vcpu_id: VCPUIndex) -> Result<()> {
		if self.vcpus.contains_key(&vcpu_id) {
			return Err(anyhow::anyhow!("vcpu {} already initialized", vcpu_id));
		}

		assert_ne!(self.elf_path, "");

		Arc::get_mut(&mut self.vcpus)
			.expect("failed to get mutable reference to vcpus")
			.insert(
				vcpu_id,
				Vcpu {
					icount: Default::default(),
					flame:  Arc::new(Mutex::new(FlameWriter::new(
						&self.elf_path,
						vcpu_id,
						self.buffer_capacity,
					)?)),
				},
			);

		Ok(())
	}

	fn on_translation_block_translate(
		&mut self,
		_id: PluginId,
		tb: TranslationBlock,
	) -> Result<()> {
		for insn in tb.instructions() {
			let vcpus = self.vcpus.clone();
			let vaddr = insn.vaddr();
			let sym = self.symtab.get(&insn.vaddr()).cloned();
			let entry_point = self.elf_entry_addr;
			let has_hit_entry_point = self.has_hit_entry_point.clone();

			assert_ne!(entry_point, 0);

			insn.register_execute_callback_flags(
				move |vcpu_idx| {
					let vcpu = vcpus
						.get(&vcpu_idx)
						.expect("instruction executed on unregistered vcpu");

					let icount = vcpu.icount.load(SeqCst);

					if vaddr == entry_point {
						has_hit_entry_point.store(true, SeqCst);
						vcpu.flame.lock().unwrap().reset(icount);
					}

					if has_hit_entry_point.load(SeqCst) {
						vcpu.flame.lock().unwrap().hit(&sym, vaddr, icount);
					}

					vcpu.icount.fetch_add(1, SeqCst);
				},
				CallbackFlags::QEMU_PLUGIN_CB_NO_REGS,
			);
		}

		Ok(())
	}
}

impl Plugin for Kflame {}

#[ctor]
fn init() {
	PLUGIN
		.set(Mutex::new(Box::new(Kflame::default())))
		.map_err(|_| anyhow::anyhow!("failed to set plugin Kflame"))
		.expect("failed to set plugin Kflame");
}
