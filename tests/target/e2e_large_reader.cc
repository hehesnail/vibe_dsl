void large_reader() {
  // Runtime arguments
  uint64_t src_dram_addr = get_arg_val<uint32_t>(0);
  uint64_t src_dram_addr_hi = get_arg_val<uint32_t>(1);
  src_dram_addr |= (src_dram_addr_hi << 32);
  uint32_t num_tiles = get_arg_val<uint32_t>(2);

  // CB configuration
  constexpr uint32_t cb_id = 0;
  constexpr uint32_t tile_size = 2048;

  // Read loop
  for (uint32_t i = 0; i < num_tiles; i++) {
    cb_reserve_back(cb_id, 1);
    uint32_t write_ptr = get_write_ptr(cb_id);
    uint64_t src_addr = src_dram_addr + i * tile_size;
    noc_async_read(src_addr, write_ptr, tile_size);
    noc_async_read_barrier();
    cb_push_back(cb_id, 1);
  }
}
