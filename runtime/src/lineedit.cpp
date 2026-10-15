#include "polima/lineedit.hpp"

#include <cstddef>
#include <csignal>
#include <cstdio>
#include <iostream>

#include <termios.h>
#include <unistd.h>

namespace polima {

volatile sig_atomic_t g_interrupted = 0;

namespace {

void on_interrupt(int) { g_interrupted = 1; }

// Raw mode for the duration of one read, restored on every exit path. ISIG is
// left ON so Ctrl-C still raises SIGINT during a long command; the editor reads
// its own byte 3 only while it has the terminal.
class RawMode {
 public:
  explicit RawMode(bool enable) : enabled_(enable) {
    if (!enabled_) return;
    if (tcgetattr(STDIN_FILENO, &saved_) != 0) { enabled_ = false; return; }
    termios raw = saved_;
    raw.c_lflag &= ~(ICANON | ECHO);
    raw.c_cc[VMIN] = 1;
    raw.c_cc[VTIME] = 0;
    // TCSADRAIN, not TCSAFLUSH: flushing would discard anything typed
    // ahead of the prompt, and a session where fast typing loses
    // characters is worse than no line editing at all.
    if (tcsetattr(STDIN_FILENO, TCSADRAIN, &raw) != 0) enabled_ = false;
  }
  ~RawMode() {
    if (enabled_) tcsetattr(STDIN_FILENO, TCSADRAIN, &saved_);
  }
  RawMode(const RawMode&) = delete;
  RawMode& operator=(const RawMode&) = delete;

 private:
  termios saved_{};
  bool enabled_;
};

size_t previous_word(const std::string& buffer, size_t cursor) {
  while (cursor > 0 && buffer[cursor - 1] == ' ') --cursor;
  while (cursor > 0 && buffer[cursor - 1] != ' ') --cursor;
  return cursor;
}

}  // namespace

void install_interrupt_handler() {
  struct sigaction action {};
  action.sa_handler = on_interrupt;
  sigemptyset(&action.sa_mask);
  action.sa_flags = SA_RESTART;
  sigaction(SIGINT, &action, nullptr);
}

LineEditor::LineEditor() : interactive_(isatty(STDIN_FILENO) == 1) {}

void LineEditor::set_completions(std::vector<std::string> words) {
  completions_ = std::move(words);
}

void LineEditor::redraw(const std::string& prompt, const std::string& buffer,
                        size_t cursor) const {
  // Carriage return, redraw, clear to end of line, then walk the cursor back.
  std::string frame = "\r" + prompt + buffer + "\x1b[K";
  const size_t tail = buffer.size() - cursor;
  if (tail > 0) frame += "\x1b[" + std::to_string(tail) + "D";
  std::fwrite(frame.data(), 1, frame.size(), stdout);
  std::fflush(stdout);
}

void LineEditor::complete(std::string& buffer, size_t& cursor) const {
  const size_t start = previous_word(buffer, cursor);
  const std::string prefix = buffer.substr(start, cursor - start);
  if (prefix.empty() && completions_.empty()) return;

  std::vector<const std::string*> matches;
  for (const auto& word : completions_)
    if (word.rfind(prefix, 0) == 0) matches.push_back(&word);
  if (matches.empty()) return;

  // One match completes; several print the options and complete the longest
  // common prefix, which is what every shell does.
  std::string insert = *matches.front();
  for (const auto* candidate : matches) {
    size_t index = 0;
    while (index < insert.size() && index < candidate->size() &&
           insert[index] == (*candidate)[index]) {
      ++index;
    }
    insert.resize(index);
  }
  if (matches.size() > 1) {
    std::string listing = "\n";
    for (const auto* candidate : matches) listing += "  " + *candidate;
    listing += "\n";
    std::fwrite(listing.data(), 1, listing.size(), stdout);
  }
  if (insert.size() > prefix.size()) {
    buffer.replace(start, cursor - start, insert);
    cursor = start + insert.size();
    if (matches.size() == 1) {
      buffer.insert(buffer.begin() + static_cast<std::ptrdiff_t>(cursor), ' ');
      ++cursor;
    }
  }
}

bool LineEditor::read(const std::string& prompt, std::string& line) {
  if (!interactive_) {
    std::cout << prompt << std::flush;
    if (!std::getline(std::cin, line)) return false;
    return true;
  }

  RawMode raw(true);
  std::string buffer;
  size_t cursor = 0;
  size_t browsing = history_.size();   // == size() means "the line being typed"
  std::string pending;                 // the in-progress line, parked while browsing

  redraw(prompt, buffer, cursor);
  while (true) {
    char key = 0;
    const ssize_t got = ::read(STDIN_FILENO, &key, 1);
    if (got <= 0) {
      if (errno == EINTR) { g_interrupted = 0; continue; }
      std::cout << "\n";
      return false;
    }

    if (key == '\r' || key == '\n') {
      std::cout << "\n" << std::flush;
      line = buffer;
      if (!buffer.empty() && (history_.empty() || history_.back() != buffer))
        history_.push_back(buffer);
      return true;
    }
    if (key == 3) {                    // Ctrl-C: abandon the line, keep the session
      std::cout << "^C\n" << std::flush;
      line.clear();
      return true;
    }
    if (key == 4) {                    // Ctrl-D: EOF only on an empty line
      if (buffer.empty()) { std::cout << "\n"; return false; }
      if (cursor < buffer.size()) buffer.erase(cursor, 1);
    } else if (key == 127 || key == 8) {
      if (cursor > 0) { buffer.erase(--cursor, 1); }
    } else if (key == 1) {             // Ctrl-A
      cursor = 0;
    } else if (key == 5) {             // Ctrl-E
      cursor = buffer.size();
    } else if (key == 21) {            // Ctrl-U
      buffer.erase(0, cursor);
      cursor = 0;
    } else if (key == 11) {            // Ctrl-K
      buffer.erase(cursor);
    } else if (key == 23) {            // Ctrl-W
      const size_t start = previous_word(buffer, cursor);
      buffer.erase(start, cursor - start);
      cursor = start;
    } else if (key == '\t') {
      complete(buffer, cursor);
    } else if (key == 27) {            // escape sequence
      char rest[2] = {0, 0};
      if (::read(STDIN_FILENO, &rest[0], 1) != 1) continue;
      if (rest[0] != '[' && rest[0] != 'O') continue;
      if (::read(STDIN_FILENO, &rest[1], 1) != 1) continue;
      switch (rest[1]) {
        case 'A':                      // up: older
          if (browsing > 0) {
            if (browsing == history_.size()) pending = buffer;
            buffer = history_[--browsing];
            cursor = buffer.size();
          }
          break;
        case 'B':                      // down: newer
          if (browsing < history_.size()) {
            ++browsing;
            buffer = browsing == history_.size() ? pending : history_[browsing];
            cursor = buffer.size();
          }
          break;
        case 'C': if (cursor < buffer.size()) ++cursor; break;
        case 'D': if (cursor > 0) --cursor; break;
        case 'H': cursor = 0; break;
        case 'F': cursor = buffer.size(); break;
        case '3': {                    // delete: consumes a trailing '~'
          char tilde = 0;
          if (::read(STDIN_FILENO, &tilde, 1) == 1 && cursor < buffer.size())
            buffer.erase(cursor, 1);
          break;
        }
        default: break;
      }
    } else if (static_cast<unsigned char>(key) >= 32) {
      buffer.insert(cursor, 1, key);
      ++cursor;
    }
    redraw(prompt, buffer, cursor);
  }
}

}  // namespace polima
