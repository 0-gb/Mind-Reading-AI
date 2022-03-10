from sklearn.naive_bayes import GaussianNB
from tkinter import Tk, Label
import random


def get_id_dictionary(allowed_symbols_local):
    vocab = sorted(set(allowed_symbols_local))
    constructed_char_2_int = {}
    constructed_int_2_char = {}
    counter = 0
    for letter in vocab:
        constructed_char_2_int[letter] = counter
        constructed_int_2_char[counter] = letter
        counter += 1
    return constructed_char_2_int, constructed_int_2_char


def preprocess_char(char):
    global correct_count
    selected_key = char_2_int[char.capitalize()]
    prediction_correct = predicted_key == selected_key
    if prediction_correct:
        correct_count += 1
    return selected_key, prediction_correct, correct_count


def step_preprocess_x_y(learn_X, learn_Y, selected_key):
    learn_X.pop(0)
    learn_Y.pop(0)
    learn_X.append(X_advanced)
    learn_Y.append(selected_key)
    return learn_X, learn_Y


def create_next_x(learn_X, selected_key, prediction_correct):
    X_advanced = learn_X[-1][1:-1]
    X_advanced.append(selected_key)
    X_advanced.append(prediction_correct)
    return X_advanced


def texts_from_metrics(selected_key, predicted_key, accuracy, window_accuracy, window_size):
    global int_2_char
    output_text1 = "Chosen - Prediction : " + int_2_char[selected_key] + " - " + int_2_char[predicted_key]
    output_text2 = "Overall accuracy: {:.2f}".format(accuracy)
    output_text3 = "Last {} guesses accuracy: {:.2f}".format(window_size, window_accuracy)
    output_text4 = ""
    if accuracy > 1 / 3:
        output_text4 += "Human successfully predicted by AI    "
    else:
        output_text4 += "Human not successfully predicted by AI"
    return [output_text1, output_text2, output_text3, output_text4]


def process_key_press(event):
    if event.char not in 'rspRSP' or len(event.char) < 1:
        return
    global case_counter, predicted_key, X_advanced, correctness_hist, learn_X, learn_Y

    selected_key, prediction_correct, accurate_guesses = preprocess_char(event.char)
    learn_X, learn_Y = step_preprocess_x_y(learn_X, learn_Y, selected_key)

    case_counter += 1
    accuracy = accurate_guesses / case_counter
    correctness_hist.append(prediction_correct)
    window_accuracy = sum(correctness_hist[-window_size:]) / min(case_counter, window_size)

    X_advanced = create_next_x(learn_X, selected_key, prediction_correct)
    output_texts = texts_from_metrics(selected_key, predicted_key, accuracy, window_accuracy, window_size)

    w1 = Label(root, text=output_texts[0], anchor="w", width=300)
    w2 = Label(root, text=output_texts[1], anchor="w", width=300)
    w3 = Label(root, text=output_texts[2], anchor="w", width=300)
    w4 = Label(root, text=output_texts[3], anchor="w", width=300)
    w1.place(x=10, y=10)
    w2.place(x=10, y=30)
    w3.place(x=10, y=50)
    w4.place(x=10, y=70)
    clf.partial_fit([learn_X[-1]] * case_counter, [learn_Y[-1]] * case_counter)
    predicted_key = clf.predict([X_advanced])[0]


allowed_symbols = 'RPS'
char_2_int, int_2_char = get_id_dictionary(allowed_symbols)
correctness_hist = []

window_size = 100
work_list = [char_2_int[element] for element in
             ','.join(random.choice(allowed_symbols) for _ in range(2 * window_size)).split(',')]

learn_X = [work_list[i:i + window_size] for i in range(0, len(work_list) - window_size)]
for i in range(0, len(work_list) - window_size):
    learn_X[i].append(random.randint(0, 1))
learn_Y = work_list[window_size:]


predicted_key = char_2_int[random.choice(allowed_symbols)]
X_advanced = create_next_x(learn_X, learn_Y[-1], predicted_key)
correct_count, case_counter = 0, 0

clf = GaussianNB()
clf.fit(learn_X, learn_Y)


root = Tk()
root.geometry('600x100')
root.bind("<Key>", process_key_press)
root.mainloop()
