import tkinter as tk
import numpy as np

def mylinfit(inputs): #to calculate a, b:
    x = list(map(lambda x: x[0], inputs))
    y = list(map(lambda x: x[1], inputs))
    num = np.sum((x - np.mean(x)) * (y - np.mean(y)))
    den = np.sum((x - np.mean(x)) ** 2)
    a = (num / den)
    b = (np.mean(y) - a * np.mean(x))
    return a, b


userInputs = [] # to store user's selected points coordinates
stopped = False


def addPoint(x, y):
    global userInputs
    userInputs.append([x, y])

def click(event): #left button
    global stopped
    if not stopped:
        x, y = event.x, event.y
        canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill="black")
        addPoint(x, y)
        print("Clicked in:", x, y)

def stop(event): #right button
    global stopped
    if not stopped:
        stopped = True
        canvas.unbind('<Button-1>')
        print("Stopped")

        a, b = mylinfit(userInputs)
        x_values = np.linspace(0, width, 500)
        y_values = a * x_values + b
        canvas.create_line(*zip(x_values, y_values), fill="red", width=3)

def draw_axes(canvas, width, height):
    origin_x = width // 2
    origin_y = height // 2

    canvas.create_line(0, origin_y, width, origin_y, fill="black")
    canvas.create_line(origin_x, 0, origin_x, height, fill="black")



root = tk.Tk()
width = 500
height = 500
canvas = tk.Canvas(root, width=500, height=500, bg="white")
canvas.pack()
draw_axes(canvas, 500, 500)
canvas.bind("<Button-1>", click)
canvas.bind("<Button-2>", stop)
root.mainloop()