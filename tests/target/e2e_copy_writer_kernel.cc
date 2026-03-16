void copy_writer() {
  // Runtime arguments
  uint64_t dst_dram_addr = get_arg_val<uint32_t>(0);
  uint64_t dst_dram_addr_hi = get_arg_val<uint32_t>(1);
  dst_dram_addr |= (dst_dram_addr_hi << 32);
  uint32_t num_tiles = get_arg_val<uint32_t>(2);

  // CB configuration
  constexpr uint32_t cb_id = 0;
  constexpr uint32_t tile_size = 2048;

  // Write loop
  for (uint32_t i = 0; i < num_tiles; i++) {
    cb_wait_front(cb_id, 1);
    uint32_t read_ptr = get_read_ptr(cb_id);
    uint64_t dst_addr = dst_dram_addr + i * tile_size;
    noc_async_write(read_ptr, dst_addr, tile_size);
    noc_async_write_barrier();
    cb_pop_front(cb_id, 1);
  }
}
