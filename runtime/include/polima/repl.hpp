// The interactive half of polima-cli.
//
// No arguments, or `--interactive`, opens a session over the DevKit's model
// store: a model loads once (weights onto the MLA) and every inference after
// that is fast. `--bundle` runs once and exits, for scripts and the deploy
// smoke test.
#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace polima {

// One entry in the DevKit's model store.
struct StoreEntry {
  std::string name;
  std::filesystem::path root;
  std::string policy;        // empty for a hand-built tree with no bundle.json
  size_t graphs = 0;
  std::string robot;         // calibration/profile id from plan.json
  std::string action_shape;  // e.g. 100x7, visibly distinguishes checkpoints
  size_t elf_bytes = 0;
  bool managed = false;      // has bundle.json -> PoLiMa built it
  bool current = false;      // the `current` symlink points here
};

// Everything in `store` that looks like a servable model tree, sorted with
// PoLiMa bundles first. Trees without a plan.json are listed but cannot be
// loaded -- the listing says so rather than failing later.
std::vector<StoreEntry> scan_store(const std::filesystem::path& store);

// Resolve a one-based list index or exact bundle name, rejecting unmanaged
// trees without changing device state.
std::string resolve_bundle(const std::filesystem::path& store,
                           const std::string& selection);

// Atomically point <store>/../current at a managed bundle selected by its
// one-based list index or exact name. Returns the selected bundle name.
std::string activate_bundle(const std::filesystem::path& store,
                            const std::string& selection);

// Run the interactive session. Returns a process exit code.
int repl(const std::filesystem::path& store, const std::string& preselect, bool verbose);

}  // namespace polima
