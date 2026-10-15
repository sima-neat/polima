// A minimal line editor: history, cursor movement, tab completion.
//
// Without this a "session" is `std::getline`, where the up arrow prints `^[[A`
// and a typo means retyping the line. That is the difference between something
// you can work in and something you tolerate.
//
// It is written out rather than linked because the board has libreadline.so.8
// but not its headers, so `-lreadline` would add an apt dependency to the
// on-board build for ~200 lines of behaviour.
//
// Falls back to std::getline when stdin is not a terminal, so piping a script
// into polima-cli works unchanged.
#pragma once

#include <csignal>
#include <string>
#include <vector>

namespace polima {

class LineEditor {
 public:
  LineEditor();

  // Reads one line. Returns false at end of input (Ctrl-D on an empty line).
  // Ctrl-C abandons the current line and returns an empty string.
  bool read(const std::string& prompt, std::string& line);

  // Words offered on Tab. Set per prompt: commands always, plus model names.
  void set_completions(std::vector<std::string> words);

  // True when the session is attached to a terminal; false when piped.
  bool interactive() const { return interactive_; }

 private:
  void redraw(const std::string& prompt, const std::string& buffer, size_t cursor) const;
  void complete(std::string& buffer, size_t& cursor) const;

  std::vector<std::string> history_;
  std::vector<std::string> completions_;
  bool interactive_ = false;
};

// Set by SIGINT while a long command runs, so `bench 500` can be abandoned
// without killing the session and losing the loaded model.
extern volatile sig_atomic_t g_interrupted;
void install_interrupt_handler();

}  // namespace polima
