import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from BILSTM_CharEmb_model import BILSTMDoubleInputModel, BILSTMCRFDoubleInputModel
from utils import save_report_to_file, get_pred_and_ground_string, get_char_indices, \
    get_char_to_index_dict, get_word_to_index_mappings, get_tag_to_index_mappings, \
    read_data_for_task, aggregate_report_pkl, read_folds
import numpy as np
import tensorflow as tf
import random
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.metrics import classification_report

def NER_driver(fold = None):

    # --------------SETTINGS-------------

    max_sentence_length = 50
    EPOCHS = 1000
    BATCH_SIZE = 256
    EMBEDDING = 40
    # pre_proc = "none"
    pre_proc = "none"
    vectorizer_model_name = 'lexical_300'
    missing_values_handled = False
    task_name = "food-classification"
    use_crf = True

    # seeds
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


    # --------------SETTINGS-------------

    if fold == None:
            full_data, train_data, test_data = read_data_for_task(task_name)
            print(full_data.shape)
            print(train_data.shape)
            print(test_data.shape)
    else:
            full_data, train_data, test_data = read_folds(task_name, fold)
            print(full_data.shape)
            print(train_data.shape)
            print(test_data.shape)

    nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()


    if use_crf:
        model_instance = BILSTMCRFDoubleInputModel()
        nn_model_name = "BILSTM_CharEmb_CRF"
    else:
        model_instance = BILSTMDoubleInputModel()
        nn_model_name = "BILSTM_CharEmb"
        

    def lemmatize_df_index(df):
        df.index = df.index.map(lambda token: lemmatizer.lemmatize(token))
        return df


    if pre_proc == "lemma":
        full_data = lemmatize_df_index(full_data)


    ## Global vocabs from full file
    word2idx, idx2word, n_words, words = get_word_to_index_mappings(full_data)
    tag2idx, idx2tag, n_tags, tags = get_tag_to_index_mappings(full_data)
    char2idx, idx2char, n_chars, chars = get_char_to_index_dict(words)

    if pre_proc == "lemma":
        train_data, test_data = lemmatize_df_index(train_data), lemmatize_df_index(test_data)


    X_tr = model_instance.process_X(train_data, word2idx, max_sentence_length)
    X_te = model_instance.process_X(test_data, word2idx, max_sentence_length)

    Y_tr, Y_tr_str = model_instance.process_Y(train_data, tag2idx, max_sentence_length, n_tags)
    Y_te, Y_te_str = model_instance.process_Y(test_data, tag2idx, max_sentence_length, n_tags)

    max_word_length = np.max([len(word) for word in train_data.index])
    X_char_tr = get_char_indices(train_data, max_word_length, max_sentence_length, char2idx)
    X_char_te = get_char_indices(test_data, max_word_length, max_sentence_length, char2idx)

    cbks = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 5, min_delta = 5e-3)

    model = model_instance.get_compiled_model(vectorizer_model_name, missing_values_handled,
                                    max_sentence_length, max_word_length,
                                    n_words, n_chars, n_tags,
                                    word2idx)

    history = model.fit([X_tr, np.array(X_char_tr).reshape((len(X_char_tr), max_sentence_length, max_word_length))],
                        Y_tr, epochs=EPOCHS, validation_split=0.1, verbose=1, batch_size=BATCH_SIZE, callbacks = [cbks])

    preds = model.predict([X_te, np.array(X_char_te).reshape((len(X_char_te), max_sentence_length, max_word_length))])

    preds_str, ground_str = get_pred_and_ground_string(Y_test=Y_te, predictions=preds, idx2tag=idx2tag)

    assert len(preds_str) == len(ground_str)
    report = classification_report(ground_str, preds_str, output_dict=True)
        
    if fold != None:
        if vectorizer_model_name != "lexical":
            report_file_name_txt = f"{task_name}_{pre_proc}_{missing_values_handled}_e{EPOCHS}_earlystop_fold={fold}_res.txt"
            report_file_name_pkl = f"{task_name}_{pre_proc}_{missing_values_handled}_e{EPOCHS}_earlystop_fold={fold}_res.pkl"
        else:
            report_file_name_txt = f"{task_name}_{pre_proc}_e{EPOCHS}_earlystop_fold={fold}_res.txt"
            report_file_name_pkl = f"{task_name}_{pre_proc}_e{EPOCHS}_earlystop_fold={fold}_res.pkl"
        save_report_to_file(report, vectorizer_model_name=vectorizer_model_name, file_name=report_file_name_txt, n_epochs = len(history.history['loss']), which_fold = fold, nn_model_name = nn_model_name)
        aggregate_report_pkl(report, vectorizer_model_name=vectorizer_model_name, file_name=report_file_name_pkl, n_epochs = len(history.history['loss']), nn_model_name = nn_model_name)

    else:
        if vectorizer_model_name != "lexical":
            report_file_name_txt = f"{task_name}_{pre_proc}_{missing_values_handled}_e{EPOCHS}_earlystop_res.txt"
            report_file_name_pkl = f"{task_name}_{pre_proc}_{missing_values_handled}_e{EPOCHS}_earlystop_res.pkl"
        else:
            report_file_name_txt = f"{task_name}_{pre_proc}_e{EPOCHS}_earlystop_res.txt"
            report_file_name_pkl = f"{task_name}_{pre_proc}_e{EPOCHS}_earlystop_res.pkl"
        save_report_to_file(report, vectorizer_model_name=vectorizer_model_name, file_name=report_file_name_txt, n_epochs = len(history.history['loss']), nn_model_name = nn_model_name)
        aggregate_report_pkl(report, vectorizer_model_name=vectorizer_model_name, file_name=report_file_name_pkl, n_epochs = len(history.history['loss']), nn_model_name = nn_model_name)

    print(report_file_name_txt)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog = "CRF_train")
    parser.add_argument("--fold", type = int, dest = "which_fold", default = None)
    args = parser.parse_args()

    NER_driver(args.which_fold)

