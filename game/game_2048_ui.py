import tkinter as tk
import random

from game.env_2048 import Env2048

# self.env_2048.size = 4
SIZE = 400
GRID_PADDING = 10
BACKGROUND_COLOR_GAME = "#92877d"
BACKGROUND_COLOR_CELL_EMPTY = "#9e948a"
BACKGROUND_COLOR_DICT = {
  2: "#eee4da", 4: "#ede0c8", 8: "#f2b179", 16: "#f59563",
  32: "#f67c5f", 64: "#f65e3b", 128: "#edcf72", 256: "#edcc61",
  512: "#edc850", 1024: "#edc53f", 2048: "#edc22e",
}
CELL_COLOR_DICT = {
  2: "#776e65", 4: "#776e65", 8: "#f9f6f2", 16: "#f9f6f2",
  32: "#f9f6f2", 64: "#f9f6f2", 128: "#f9f6f2", 256: "#f9f6f2",
  512: "#f9f6f2", 1024: "#f9f6f2", 2048: "#f9f6f2",
}
FONT = ("Verdana", 24, "bold")


class Game2048UI(tk.Frame):
  def __init__(self, env_2048, *, player_mode=True):
    tk.Frame.__init__(self)
    self.env_2048 = env_2048
    self.player_mode = player_mode

    self.grid()
    self.master.title('2048')
    self.master.resizable(False, False)
    self.grid_cells = []
    self.score = 0
    self.init_grid()
    self.update_grid_cells()
    self.bind_all("<Key>", self.key_down)
    if self.player_mode:
      self.mainloop()

  def reset_env(self, env):
    self.env_2048 = env
    self.update_grid_cells()

  def init_grid(self):
    self.grid_cells.clear()
    # 计分板放到方格上方
    self.score_label = tk.Label(self, text=f"Score: {self.score}", font=(
      "Verdana", 18, "bold"), bg=BACKGROUND_COLOR_GAME, fg="#f9f6f2")
    self.score_label.grid(row=0, column=0, columnspan=self.env_2048.size, pady=(0, 0))
    background = tk.Frame(self, bg=BACKGROUND_COLOR_GAME,
                          width=SIZE, height=SIZE + 50)
    background.grid()
    # self.score_label = tk.Label(self, text=f"Score: {self.score}", font=(
    #   "Verdana", 18, "bold"), bg=BACKGROUND_COLOR_GAME, fg="#f9f6f2")
    # self.score_label.grid(row=0, column=0, columnspan=self.env_2048.size)
    for i in range(self.env_2048.size):
      grid_row = []
      for j in range(self.env_2048.size):
        cell = tk.Frame(
          background,
          bg=BACKGROUND_COLOR_CELL_EMPTY,
          width=SIZE / self.env_2048.size,
          height=SIZE / self.env_2048.size
        )
        cell.grid(row=i, column=j, padx=GRID_PADDING, pady=GRID_PADDING)
        t = tk.Label(
          master=cell,
          text="",
          bg=BACKGROUND_COLOR_CELL_EMPTY,
          justify=tk.CENTER,
          font=FONT,
          width=4,
          height=2
        )
        t.grid()
        grid_row.append(t)
      self.grid_cells.append(grid_row)

  def update_grid_cells(self):
    matrix = self.env_2048.matrix
    for i in range(self.env_2048.size):
      for j in range(self.env_2048.size):
        value = matrix[i][j]
        if value == 0:
          self.grid_cells[i][j].configure(
            text="", bg=BACKGROUND_COLOR_CELL_EMPTY)
        else:
          self.grid_cells[i][j].configure(
            text=str(value),
            bg=BACKGROUND_COLOR_DICT.get(value, "#3c3a32"),
            fg=CELL_COLOR_DICT.get(value, "#f9f6f2")
          )
    self.score_label.config(text=f"Score: {self.score}")
    self.update_idletasks()

  def key_down(self, event):
    key = event.keysym
    if key in ('Up', 'Down', 'Left', 'Right'):
      _, score_gain, done, _ = self.env_2048.step(key)
      self.score += score_gain
      self.update_grid_cells()
      if done and self.player_mode:
          self.game_over()

  def game_over(self):
    over = tk.Toplevel(self)
    over.title("Game Over")
    tk.Label(over, text=f"Game Over!\nFinal Score: {self.score}", font=(
      "Verdana", 18, "bold")).pack(padx=20, pady=20)
    tk.Button(over, text="Restart", command=lambda: [
              over.destroy(), self.restart()]).pack(pady=10)
    tk.Button(over, text="Quit", command=self.quit).pack(pady=5)

  def restart(self):
    self.score = 0
    self.env_2048.reset()
    self.update_grid_cells()


if __name__ == '__main__':
  Game2048UI(
    Env2048(),
    # mainloop=False,
  )
