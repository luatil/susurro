from context import Context
from gui import GUI

def main() -> None:
    ctx = Context()
    gui = GUI(ctx)
    gui.render()

if __name__ == "__main__":
    main()