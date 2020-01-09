import tkinter as tk
import numpy as np
from tkinter import *
from tkinter import ttk
from tkinter.filedialog import askopenfilename
from tkinter import messagebox
from PIL import Image, ImageTk
from PredictionModule import KNearNeighborsPrediction
from SongFeatureExtractor import song_processing
import warnings

# the apps tkinter main Style attribute
LARGE_FONT = ("Verdana", 20)
FRAMESIZE = "1000x700"
STYLEFILESFOLDER = "StyleFiles"


class MainApplication(tk.Tk):
    """
    this class is MightyFi main app.
    it wraps a tkinter application and runs tkinter frames.
    it runs and manages all the pages.
    it instansiate all pages and lunches the Entry page
    """
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.geometry(FRAMESIZE)
        self.iconbitmap('{0}/equalizer.ico'.format(STYLEFILESFOLDER))

        tk.Tk.iconbitmap(self)
        tk.Tk.wm_title(self, "MigHtyFi")
        container = ttk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (EntryPage, SongAnalyzer, AboutPage):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(EntryPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


class EntryPage(tk.Frame):
    """
    the application "Home Page"
    """
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg="#211F1E")
        self.parent = parent
        self.controller = controller

        label = ttk.Label(self, text="Welcome To MigHtYFi!", font=LARGE_FONT)
        label.grid(row=0, column=0, columnspan=2)

        main_photo = Image.open("{0}/spectrum_image.png".format(STYLEFILESFOLDER))
        main_photo = ImageTk.PhotoImage(main_photo)
        photo_label = ttk.Label(self, image=main_photo)
        label.image = main_photo
        photo_label.grid(row=1, column=0, columnspan=2)
        button_to_home_page = ttk.Button(self, text="Go To Song Analyzer", command=lambda: self.controller.show_frame(SongAnalyzer))
        button_to_home_page.grid(row=3, column=1)
        button_to_about_page = ttk.Button(self, text="About",
                                          command=lambda: self.controller.show_frame(AboutPage))
        button_to_about_page.grid(row=3, column=0)


class AboutPage(tk.Frame):
    """
    An about page about us and our project
    """
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg="#363332")
        label = ttk.Label(self, text="About Page", font=LARGE_FONT)
        label.grid(row=0, column=0)

        label_about_us = tk.Text(self, width=100, height=20)
        label_about_us.insert('1.0', open("{0}/aboutmightyfi.txt".format(STYLEFILESFOLDER)).read())
        label_about_us.grid(row=1, column=0, columnspan=3)
        button1 = ttk.Button(self, text="Home Page",
                             command=lambda: controller.show_frame(EntryPage))
        button1.grid(row=2, column=1)

        button2 = ttk.Button(self, text="To Our Song Analyzer",
                             command=lambda: controller.show_frame(SongAnalyzer))
        button2.grid(row=2, column=2)


class SongAnalyzer(tk.Frame):
    """
    the page to get the user input. if input is ok runs it through our model.
    shows the songs lyrics and true number of likes on youtube if available.
    this class uses messageboxes to alert the user to scenarios.
    """
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg="#211F1E")

        Label(self, text='Song Path').grid(row=0)
        Label(self, text='Song Name').grid(row=1)
        Label(self, text='Artist name').grid(row=2)

        self.entry_song_path = Entry(self)
        self.entry_song_name = Entry(self)
        self.entry_artist_name = Entry(self)

        self.entry_song_path.grid(row=0, column=1)
        self.entry_song_name.grid(row=1, column=1)
        self.entry_artist_name.grid(row=2, column=1)

        b1 = Button(self, text='Browse Song File', command=self.browse_button)
        b1.grid(row=0, column=2, sticky=W, pady=4)

        self.lyrics_text = tk.Text(self, width=40, height=20)
        self.lyrics_text.insert('1.0', "Song Lyrics:\n")
        # making the lyrics text box read only
        self.lyrics_text.config(state=DISABLED)
        self.lyrics_text.grid(row=3, column=0, columnspan=3)

        Label(self, text='\t'*4, bg='#211F1E').grid(row=3, column=4)

        b2 = Button(self, text='Predict', command=self.predict_song_success, bg='#11ba25', font=('calibri', 32), fg='white')
        b2.grid(row=3, column=5, sticky=W, pady=4)

        self.true_y_of_song_textbox = tk.Text(self, width=40, height=5, state=DISABLED)
        self.true_y_of_song_textbox.grid(row=4, column=0, columnspan=3)
        # making the true y axis result text box read only
        self.true_y_of_song_textbox.config(state=DISABLED)

        self.predicted_y_of_song_textbox = tk.Text(self, width=40, height=5, state=DISABLED)
        self.predicted_y_of_song_textbox.grid(row=5, column=0, columnspan=3)
        # making the true y axis result text box read only
        self.predicted_y_of_song_textbox.config(state=DISABLED)

        self.back_home = lambda: controller.show_frame(EntryPage)
        b2 = Button(self, text='Back Home', command=self.clean_and_back_home)
        b2.grid(row=6, column=0, columnspan=3)

    def clean_and_back_home(self):
        self.entry_song_path.delete(0, END)
        self.entry_song_name.delete(0, END)
        self.entry_artist_name.delete(0, END)
        self.lyrics_text.config(state=NORMAL)
        self.lyrics_text.delete('2.0', END)
        self.lyrics_text.config(state=DISABLED)
        self.true_y_of_song_textbox.config(state=NORMAL)
        self.true_y_of_song_textbox.delete('1.0', END)
        self.true_y_of_song_textbox.config(state=DISABLED)
        self.predicted_y_of_song_textbox.config(state=NORMAL)
        self.predicted_y_of_song_textbox.delete('1.0', END)
        self.predicted_y_of_song_textbox.config(state=DISABLED)
        self.back_home()

    def browse_button(self):
        """
        opens and retrieves the input of the file path
        :return:
        """
        self.entry_song_path.delete(0, END)
        self.entry_song_path.insert(0, askopenfilename(initialdir='/', title='Select File'))

    def predict_song_success(self):
        """
        this function shows the user the result of the prediction model on hers/his input
        message boxes are used to alert the user.
        the features are extracted via the ...Extractor classes.
        it fills in nan values if needed for the features of the analyzed song alone with the medians of our
        dataset (hard coded).
        it uses the PredictionModule.KNN.. class to train a knn model and classify the analyzed song.
        it shows the user the prediction in a text box under the true Y extracted from Youtube.
        :return:
        """
        if self.entry_song_path.get() == "" or self.entry_artist_name.get() == "" or self.entry_song_name.get() == "":
            messagebox.showinfo("Wrong input", "All input must be filled!")
            return
        messagebox.showinfo("Please wait...", "Analyzing song {0}".format(self.entry_song_name.get()))
        song_dict = self.get_feature()
        # print(song_dict)
        self.lyrics_text.config(state=NORMAL)
        self.true_y_of_song_textbox.config(state=NORMAL)
        if song_dict is not None:
            if song_dict['lyrics'] is not None:
                self.lyrics_text.insert('2.0', song_dict['lyrics'])
            else:
                self.lyrics_text.insert('2.0', "Sorry, no lyrics were found :-( ")

            self.true_y_of_song_textbox.insert('1.0', "True Y: {0}".format(song_dict['y']))
            X_test = song_dict['features']
            if X_test[-1] is None:
                # fill with median values
                X_test[len(X_test)-4:] = [0.080717489, 0.03125, 0.429372, 0.9359]

            messagebox.showinfo("Please wait...", "Now predicting song success!")
        else:
            messagebox.showinfo("oh no", "Something went wrong!")

        # disabling the text boxes to read only
        self.true_y_of_song_textbox.config(state=DISABLED)
        self.lyrics_text.config(state=DISABLED)

        try:
            classifier = KNearNeighborsPrediction()
            prediction = classifier.predict(np.reshape(X_test, (1, 14)))
            self.predicted_y_of_song_textbox.config(state=NORMAL)
            self.predicted_y_of_song_textbox.insert('1.0', "Y Prediction =".format(prediction))
            self.predicted_y_of_song_textbox.insert('2.0', "{0}".format(prediction))
            self.predicted_y_of_song_textbox.config(state=DISABLED)
        except Exception as e:
            messagebox.showinfo("Something went wrong", e)

    def get_feature(self):
        """
        uses the song processing model to retrieve data from youtube, lyrics and lyrics features.
        it also uses the SongFeatureExtractor class to process the file that was given via the GUI,
        and retrieve the musical features.
        :return: a dictionary with song features, song like number on youtube, lyrics etc. .
        """
        return song_processing(self.entry_song_path.get(), self.entry_song_name.get(), self.entry_artist_name.get())


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        app = MainApplication()
        app.mainloop()

# if we only had enough sample with a DNN
# class SongAnalyserPage(tk.Frame):
#
#     def __init__(self, parent, controller):
#         tk.Frame.__init__(self, parent, bg="#211F1E")
#         self.parent = parent
#         self.file_path = ""
#         self.file_path_to_trained_model = ""
#         self.pred_model = None
#         self.data_processor = DataPreprocessor()
#
#         label = ttk.Label(self, text="Start Page", font=LARGE_FONT)
#         label.grid(row=0, column=0, columnspan=3)
#
#         # main photo
#         main_photo = Image.open("{0}/spectrum_image.png".format(STYLEFILESFOLDER))
#         main_photo = ImageTk.PhotoImage(main_photo)
#         photo_label = ttk.Label(self, image=main_photo)
#         label.image = main_photo
#         photo_label.grid(row=1, column=0, columnspan=2)
#         # lyrics text box
#         self.lyrics_text = tk.Text(self, width=20, height=20)
#         self.lyrics_text.insert('1.0', "Song Lyrics:\n")
#         self.lyrics_text.grid(row=1, column=2, columnspan=1)
#         # load model button
#         button_to_open_files = ttk.Button(self, text="Load Model", command=self.load_model_clicked)
#         button_to_open_files.grid(row=2, column=3)
#
#         button_to_open_files = ttk.Button(self, text="Open files", command=self.open_file_clicked)
#
#         button_to_open_files.grid(row=2, column=1)
#
#         button_to_analyze_file = ttk.Button(self, text="Analyze File", command=self.analyse_song_clicked)
#         button_to_analyze_file.grid(row=2, column=0)
#
#         button2 = ttk.Button(self, text="Home Page",
#                              command=lambda: controller.show_frame(EntryPage))
#         button2.grid(row=2, column=2)
#
#     def open_file_clicked(self):
#         name = askopenfilename(initialdir="C:/Users/Batman/Documents/Programming/tkinter/",
#                                filetypes=(("m4a file", "*.m4a"), ("mp3 file", "*.mp3"), ("Wave File", "*.wav"), ("All Files", "*.*")),
#                                title="Choose a file."
#                                )
#         self.file_path = name
#
#     def load_model_clicked(self):
#         name = askopenfilename(initialdir="C:/Users/Batman/Documents/Programming/tkinter/",
#                                filetypes=(("SAV file", "*.sav"), ("All Files", "*")),
#                                title="Choose a file.")
#         self.file_path_to_trained_model = name
#
#     def analyse_song_clicked(self):
#         if self.file_path == "":
#             messagebox.showinfo("ERROR!", "no file was selected!")
#             return
#         self.pred_model = PredictionModule(trained_already=self.file_path_to_trained_model)
#         self.lyrics_text.insert(tk.END, "again")
#
#         messagebox.showinfo("Thank you", "Your song {0} is analysed\nA browser window will be open with the results".format(self.file_path[:-4]))