import tkinter as tk
from tkinter import *
from tkinter.filedialog import askopenfilename, askdirectory
from PIL import Image, ImageTk
import os
import warnings
import backend
import matplotlib.pyplot as plt



FONTS = {'large': ('calibri', 54, 'bold', 'underline'), 'button_l': ('calibri', 24, 'bold'),
         'button_s': ('calibri', 14, 'bold'), 'text': ('calibri', 14)}
FRAME_SIZE = '1000x700'
RESOURCES_DIR_PATH = os.path.join(os.getcwd(), 'resources')
GRAPHS_DIR_PATH = os.path.join(os.getcwd(), 'graphs')
COLORS = {'frame_bg': '#99ceff', 'button_bg': '#0938e3', 'button_fg': '#ffffff', 'label_fg': '#00249c',
          'browse_bg': '#939499', 'browse_fg': '#000000', 'predict': '#1aad03'}
plot_num = 0


class MainApp(tk.Tk):
    """
    Main screen of the app
    can decide if train or test
    """

    def __init__(self, *args, **kwargs):
        super(MainApp, self).__init__(*args, **kwargs)
        self.geometry(FRAME_SIZE)
        # self.iconbitmap(os.path.join(RESOURCES_DIR_PATH, 'logo.png'))
        # tk.Tk.iconbitmap(self)
        tk.Tk.wm_title(self, 'Decision Making')

        containter = Frame(self)
        containter.pack(side='top', fill='both', expand=True)
        containter.grid_rowconfigure(0, weight=1)
        containter.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in [HomeFrame, TestFrame, TrainFrame]:
            self.frames[F] = F(containter, self)
            self.frames[F].grid(row=0, column=0, sticky='nsew')

        self.display_frame(HomeFrame)

    def display_frame(self, frame_key):
        self.frames[frame_key].tkraise()


class HomeFrame(tk.Frame):
    """
    Home Page Frame!
    """

    def __init__(self, parent, controller):
        super(HomeFrame, self).__init__(parent, bg=COLORS['frame_bg'], width=50, height=50)
        self.parent = parent
        self.controller = controller

        # Welcome Title
        label = Label(self, text='Welcome', font=FONTS['large'], bg=COLORS['frame_bg'], fg=COLORS['button_bg'])
        label.grid(row=0, column=0, padx=375, sticky=W, columnspan=3)

        # Logo
        load = Image.open(os.path.join(RESOURCES_DIR_PATH, 'logo.png'))
        load = load.resize((300, 300))
        render = ImageTk.PhotoImage(load)
        img = Label(self, image=render)
        img.image = render
        img.grid(row=1, column=0, padx=350, pady=30, sticky=W, columnspan=3)

        # Buttons
        btn_train_kwargs = {'text': 'Train Model', 'command': lambda: self.controller.display_frame(TrainFrame),
                            'bg': COLORS['button_bg'], 'fg': COLORS['button_fg'], 'font': FONTS['button_l']}
        btn_test_kwargs = {'text': 'Test  Model', 'command': lambda: self.controller.display_frame(TestFrame),
                           'bg': COLORS['button_bg'], 'fg': COLORS['button_fg'], 'font': FONTS['button_l']}
        btn_train = Button(self, **btn_train_kwargs)
        btn_train.grid(row=2, column=0, pady=50, sticky=E)
        btn_test = Button(self, **btn_test_kwargs)
        btn_test.grid(row=2, column=2, pady=50, sticky=W)


class SystemFrame(tk.Frame):
    def __init__(self, parent, controller):
        super(SystemFrame, self).__init__(parent, bg=COLORS['frame_bg'])
        self.parent = parent
        self.controller = controller
        self.entries = []
        self.checkbuttons = []

    def clean_and_home(self):
        for entry in self.entries:
            entry.delete(0, END)
        for check in self.checkbuttons:
            check.deselect()
        self.controller.display_frame(HomeFrame)


class TestFrame(SystemFrame):
    def __init__(self, parent, controller):
        super(TestFrame, self).__init__(parent, controller)

        # Logo
        load = Image.open(os.path.join(RESOURCES_DIR_PATH, 'logo.png'))
        load = load.resize((100, 100))
        render = ImageTk.PhotoImage(load)
        img = Label(self, image=render)
        img.image = render
        img.grid(row=0, column=0, sticky=W)

        # Test Title
        label = Label(self, text='Test Model', font=FONTS['large'], bg=COLORS['frame_bg'], fg=COLORS['button_bg'])
        label.grid(row=0, column=2, sticky=W, columnspan=4, padx=40)

        # **** Labels: Entries ****
        # load model
        lbl_load_model_kwargs = {'text': 'Load Model:', 'font': FONTS['text'], 'bg': COLORS['frame_bg'],
                                 'fg': COLORS['label_fg']}
        Label(self, **lbl_load_model_kwargs).grid(row=1, column=2, sticky=E)
        entry_load_model = Entry(self)
        entry_load_model.grid(row=1, column=3, sticky=W)
        self.entries.append(entry_load_model)  # [0]
        btn_load_model_kwargs = {'text': 'Browse', 'font': FONTS['text'], 'bg': COLORS['browse_bg'],
                                 'fg': COLORS['browse_fg'], 'width': 10, 'command': self.browse_model}
        btn_browse_model = Button(self, **btn_load_model_kwargs)
        btn_browse_model.grid(row=1, column=4, sticky=W, padx=10)
        # load test csv
        lbl_load_csv_kwargs = {'text': 'Load Data:', 'font': FONTS['text'], 'bg': COLORS['frame_bg'],
                               'fg': COLORS['label_fg']}
        Label(self, **lbl_load_csv_kwargs).grid(row=2, column=2, sticky=E)
        entry_load_csv = Entry(self)
        entry_load_csv.grid(row=2, column=3, sticky=W)
        self.entries.append(entry_load_csv)  # [1]
        btn_load_csv_kwargs = {'text': 'Browse', 'font': FONTS['text'], 'bg': COLORS['browse_bg'],
                               'fg': COLORS['browse_fg'], 'width': 10, 'command': self.browse_data}
        btn_browse_csv = Button(self, **btn_load_csv_kwargs)
        btn_browse_csv.grid(row=2, column=4, sticky=W, padx=10)
        # indexes
        lbl_indexes_kwargs = {'text': 'Question Indexes:', 'font': FONTS['text'], 'bg': COLORS['frame_bg'],
                              'fg': COLORS['label_fg']}
        Label(self, **lbl_indexes_kwargs).grid(row=3, column=2, sticky=E)
        entry_indexes = Entry(self)
        entry_indexes.grid(row=3, column=3, sticky=W)
        self.entries.append(entry_indexes)  # [2]
        # number of groups
        lbl_group_devision_kwargs = {'text': 'Groups Number:', 'font': FONTS['text'],
                                     'bg': COLORS['frame_bg'], 'fg': COLORS['label_fg']}
        Label(self, **lbl_group_devision_kwargs).grid(row=4, column=2, sticky=E)
        entry_groups_num = Entry(self)
        entry_groups_num.grid(row=4, column=3, sticky=W)
        self.entries.append(entry_groups_num)  # [3]
        # vs majority
        lbl_vs_majority_kwargs = {'text': 'Test VS Majority Rule', 'font': FONTS['text'], 'bg': COLORS['frame_bg'],
                                  'fg': COLORS['label_fg']}
        Label(self, **lbl_vs_majority_kwargs).grid(row=5, column=2, sticky=E)
        self.vs_majority = IntVar()
        cb_majority = Checkbutton(self, variable=self.vs_majority, bg=COLORS['frame_bg'])
        cb_majority.grid(row=5, column=3, sticky=W, pady=4)
        self.checkbuttons.append(cb_majority)  # [0]

        # Buttons
        # predict
        btn_predict_kwargs = {'text': 'Predict', 'command': self.predict, 'bg': COLORS['predict'],
                              'fg': COLORS['button_fg'], 'font': FONTS['button_l']}
        btn_predict = Button(self, **btn_predict_kwargs)
        btn_predict.grid(row=6, column=3, pady=10)
        # back home
        btn_home_kwargs = {'text': 'Back Home', 'command': self.clean_and_home,
                           'bg': COLORS['button_bg'], 'fg': COLORS['button_fg'], 'font': FONTS['button_s']}
        btn_back_home = Button(self, **btn_home_kwargs)
        btn_back_home.grid(row=7, column=3, pady=50, sticky=W)

    def predict(self):
        backend_kwargs = {'numOfGroups': int(self.entries[3].get()), 'file': self.entries[1].get(),
                          'modelPath': self.entries[0].get(),
                          'labels': list(map(lambda x: int(x), self.entries[2].get().replace(', ', ',').split(',')))}
        # indexes = self.entries[2].get()
        # indexes = indexes.replace(', ', ',')
        # index_list = indexes.split(',')
        # backend_kwargs['labels'] = list(map(lambda x: int(x), index_list))
        self.pop_ans(backend.checkPredictionsValues(**backend_kwargs))

    def pop_ans(self, predictions):
        popup = TestPopup(predictions)
        popup.mainloop()

    def browse_model(self):
        self.entries[0].delete(0, END)
        self.entries[0].insert(0, askopenfilename(initialdir='/', title='Select File'))

    def browse_data(self):
        self.entries[1].delete(0, END)
        self.entries[1].insert(0, askopenfilename(initialdir='/', title='Select File'))


class TrainFrame(SystemFrame):
    def __init__(self, parent, controller):
        super(TrainFrame, self).__init__(parent, controller)

        # Logo
        load = Image.open(os.path.join(RESOURCES_DIR_PATH, 'logo.png'))
        load = load.resize((100, 100))
        render = ImageTk.PhotoImage(load)
        img = Label(self, image=render)
        img.image = render
        img.grid(row=0, column=0, sticky=W)

        # Test Title
        label = Label(self, text='Train Model', font=FONTS['large'], bg=COLORS['frame_bg'], fg=COLORS['button_bg'])
        label.grid(row=0, column=2, sticky=W, columnspan=4, padx=40)

        # **** Labels: Entries ****
        # load dataset
        lbl_load_dataset_kwargs = {'text': 'Load Dataset:', 'font': FONTS['text'], 'bg': COLORS['frame_bg'],
                                   'fg': COLORS['label_fg']}
        Label(self, **lbl_load_dataset_kwargs).grid(row=1, column=2, sticky=E)
        entry_load_dataset = Entry(self)
        entry_load_dataset.grid(row=1, column=3, sticky=W)
        self.entries.append(entry_load_dataset)
        btn_load_dataset_kwargs = {'text': 'Browse', 'font': FONTS['text'], 'bg': COLORS['browse_bg'],
                                   'fg': COLORS['browse_fg'], 'width': 10, 'command': self.browse_dataset}
        btn_browse_dataset = Button(self, **btn_load_dataset_kwargs)
        btn_browse_dataset.grid(row=1, column=4, sticky=W, padx=10)
        # load test csv
        lbl_browse_save_model_kwargs = {'text': 'Save Model At:', 'font': FONTS['text'], 'bg': COLORS['frame_bg'],
                                        'fg': COLORS['label_fg']}
        Label(self, **lbl_browse_save_model_kwargs).grid(row=2, column=2, sticky=E)
        entry_save_model = Entry(self)
        entry_save_model.grid(row=2, column=3, sticky=W)
        self.entries.append(entry_save_model)
        btn_save_model_kwargs = {'text': 'Browse', 'font': FONTS['text'], 'bg': COLORS['browse_bg'],
                                 'fg': COLORS['browse_fg'], 'width': 10, 'command': self.save_model}
        btn_save_model = Button(self, **btn_save_model_kwargs)
        btn_save_model.grid(row=2, column=4, sticky=W, padx=10)
        # model name
        lbl_model_name_kwargs = {'text': 'Model\'s Name:', 'font': FONTS['text'], 'bg': COLORS['frame_bg'],
                                 'fg': COLORS['label_fg']}
        Label(self, **lbl_model_name_kwargs).grid(row=3, column=2, sticky=E)
        entry_model_name = Entry(self)
        entry_model_name.grid(row=3, column=3, sticky=W)
        self.entries.append(entry_model_name)
        # indexes
        lbl_indexes_kwargs = {'text': 'Question Indexes:', 'font': FONTS['text'], 'bg': COLORS['frame_bg'],
                              'fg': COLORS['label_fg']}
        Label(self, **lbl_indexes_kwargs).grid(row=4, column=2, sticky=E)
        entry_indexes = Entry(self)
        entry_indexes.grid(row=4, column=3, sticky=W)
        self.entries.append(entry_indexes)

        # Buttons
        # train
        btn_predict_kwargs = {'text': 'Train', 'command': self.train, 'bg': COLORS['predict'],
                              'fg': COLORS['button_fg'], 'font': FONTS['button_l']}
        btn_predict = Button(self, **btn_predict_kwargs)
        btn_predict.grid(row=5, column=3, pady=10)
        # back home
        btn_home_kwargs = {'text': 'Back Home', 'command': self.clean_and_home,
                           'bg': COLORS['button_bg'], 'fg': COLORS['button_fg'], 'font': FONTS['button_s']}
        btn_back_home = Button(self, **btn_home_kwargs)
        btn_back_home.grid(row=6, column=3, pady=50, sticky=W)

    def train(self):
        self.entries
        print('Train is not implemented')

    def browse_dataset(self):
        self.entries[0].delete(0, END)
        self.entries[0].insert(0, askopenfilename(initialdir='/', title='Select File'))

    def save_model(self):
        self.entries[1].delete(0, END)
        self.entries[1].insert(0, askdirectory(initialdir='/', title='Select File'))


class TestPopup(tk.Tk):
    def __init__(self, predictions):
        super(TestPopup, self).__init__()
        self.configure(background=COLORS['frame_bg'])
        self.geometry('600x600')
        self.wm_title('Prediction')

        titles = list(predictions[0].keys())
        titles_dict = {}
        col = 0
        for title in titles:
            titles_dict[title] = col
            lbl_title_kwargs = {'text': title, 'font': FONTS['text'], 'bg': COLORS['browse_fg'],
                                'fg': COLORS['button_fg']}
            Label(self, **lbl_title_kwargs).grid(row=0, column=col, padx=5)
            col += 1

        row = 1
        for prediction in predictions:
            for title in titles:
                lbl_prediction_kwargs = {'text': prediction[title], 'font': FONTS['text'], 'bg': COLORS['browse_fg'],
                                         'fg': COLORS['button_fg']}
                Label(self, **lbl_prediction_kwargs).grid(row=row, column=titles_dict[title], padx=5, pady=5)
            row += 1

        majority_correct = len(list(filter(lambda x: x['majority'] == x['ground_true'], predictions)))
        model_correct = len(list(filter(lambda x: x['model'] == x['ground_true'], predictions)))
        lbl_correction = {'text': 'Total', 'font': FONTS['text'], 'bg': COLORS['browse_fg'], 'fg': COLORS['button_fg']}
        Label(self, **lbl_correction).grid(row=row, column=0)

        lbl_correction = {'text': '{maj}/{tot}'.format(maj=majority_correct, tot=len(predictions)),
                          'font': FONTS['text'], 'bg': COLORS['browse_fg'], 'fg': COLORS['button_fg']}
        Label(self, **lbl_correction).grid(row=row, column=1)

        lbl_correction = {'text': '{mod}/{tot}'.format(mod=model_correct, tot=len(predictions)),
                          'font': FONTS['text'], 'bg': COLORS['browse_fg'], 'fg': COLORS['button_fg']}
        Label(self, **lbl_correction).grid(row=row, column=2)



        x_axis = ['Majority Rule', 'Our Model']
        y_axis = [majority_correct/len(predictions), model_correct/len(predictions)]
        plt.bar(x_axis, height=y_axis)
        global plot_num
        img_path = os.path.join(GRAPHS_DIR_PATH, 'plot%d.png' % plot_num)
        plot_num += 1
        plt.savefig(img_path)

        # btn_save_graph_kwargs = {'text': 'Save Graph', 'command': self.save_graph(x_axis, y_axis), 'bg': COLORS['predict'],
        #                          'fg': COLORS['button_fg'], 'font': FONTS['button_l']}
        # btn_save_graph = Button(self, **btn_save_graph_kwargs)
        # btn_save_graph.grid(row=row, column=int(col/2), pady=10)


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        app = MainApp()
        app.mainloop()
