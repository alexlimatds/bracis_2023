import json, ast
import pandas as pd

class FourRolesDM:
    def raw_to_dic_docs(self, raw_data):
        '''
        Converts JSON to dictionaries with pandas dataframes.
        '''
        n_sentences = 0
        dic_ = {}
        for json_file in raw_data:
            for doc_id, values in json_file.items(): # iterates documents in current json file
                sentences = values['sentences']
                if type(sentences) is not list:
                    sentences = ast.literal_eval(sentences)
                labels = values['complete']
                if type(labels) is not list:
                    labels = ast.literal_eval(labels)
                assert len(sentences) == len(labels)
                df = pd.DataFrame(list(zip(sentences, labels)), columns=['sentence', 'label'])
                # Converting labels
                df.loc[df['label'] == 'Issue', 'label'] = 'Fact'
                df.loc[~df['label'].isin(['Fact', 'RulingByPresentCourt', 'RatioOfTheDecision']), 'label'] = 'Other'
                dic_[doc_id] = df
                n_sentences += len(sentences)
        return dic_, n_sentences

    def load_raw_data(self, path):
        '''
        Loads data from JSON files.
        '''
        with open(path) as data_file:
            raw_data = json.load(data_file)
        return raw_data

    def get_labels_to_idx(self):
        labels_to_idx = {}
        labels_to_idx['Fact'] = 0
        labels_to_idx['RulingByPresentCourt'] = 1
        labels_to_idx['RatioOfTheDecision'] = 2
        labels_to_idx['Other'] = 3
        return labels_to_idx

    def get_valid_labels(self, labels_to_idx):
        n_valid = 0
        for idx in labels_to_idx.values():
            if idx >= 0:
                n_valid += 1
        labels = [None] * n_valid
        for l, idx in labels_to_idx.items():
            if idx >= 0:
                labels[idx] = l
        return labels

    def get_data(self):
        dataset_dir = 'malik/'
        # there are two legal domains: CL and IT
        # CL
        raw_data_cl_train = self.load_raw_data(dataset_dir + 'CL_train.json')
        raw_data_cl_dev = self.load_raw_data(dataset_dir + 'CL_dev.json')
        raw_data_cl_test = self.load_raw_data(dataset_dir + 'CL_test.json')
        # IT
        raw_data_it_train = self.load_raw_data(dataset_dir + 'IT_train.json')
        raw_data_it_dev = self.load_raw_data(dataset_dir + 'IT_dev.json')
        raw_data_it_test = self.load_raw_data(dataset_dir + 'IT_test.json')

        docs_dic_train, _ = self.raw_to_dic_docs([raw_data_cl_train, raw_data_it_train])
        docs_dic_dev, _ = self.raw_to_dic_docs([raw_data_cl_dev, raw_data_it_dev])
        docs_dic_test, _ = self.raw_to_dic_docs([raw_data_cl_test, raw_data_it_test])

        return docs_dic_train, docs_dic_dev, docs_dic_test

def main():
    # A simple test
    o = FourRolesDM()
    train, _, _ = o.get_data()
    labels = set()
    for _, df in train.items():
        labels.update(df['label'].unique())
    for l in labels:
        print(l)
    # another test
    print(o.get_valid_labels(o.get_labels_to_idx()))

if __name__ == '__main__':
    main()
