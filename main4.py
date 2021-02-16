from os import *

import tk as tk

root = tk.Tk()
root.title("Open file")
os_str_var = tk.StringVar(root)


def get_value_os():
    try:
        print(os_str_var.get())
        tk.Label(root,text=f"{os_str_var.get()}").pack()
        get_os_input = os_str_var.get()
        startfile(f"{get_os_input}")
        tk.Label(root, text=f"Good News! The system can find the path({get_os_input})
        specified. ").pack()
        print(f"Good News! The system can find the path({get_os_input}) specified. ")
        return get_os_input
    except:
        tk.Label(root,text="The system cannot find the path specified. ").pack()
        print("The system cannot find the path specified. ")


os_entry = tk.Entry(root,textvariable=os_str_var).pack()
os_button = tk.Button(root,text="Open",command=get_value_os).pack()
root.mainloop()