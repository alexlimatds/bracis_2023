import data_manager
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

labels_to_acronym = {
  'Fact' :  'FAC', 
  'Issues' : 'ISS', 
  'ArgumentPetitioner' : 'ARG-P', 
  'ArgumentRespondent' : 'ARG-R', 
  'Argument' : 'ARG', 
  'Statute' : 'STA', 
  'Dissent' : 'DIS', 
  'PrecedentReliedUpon' : 'PRE-R', 
  'PrecedentNotReliedUpon' : 'PRE-NR', 
  'PrecedentOverruled' : 'PRE-O', 
  'Precedent' : 'PRE', 
  'RulingByLowerCourt' : 'RLC', 
  'RulingByPresentCourt' : 'RPC', 
  'RatioOfTheDecision' : 'ROD', 
  'None' : 'NON'
}

def count(dic_doc, data_loader):
  # merging dataframes
  temp = []
  for id, df in dic_doc.items():
      temp.append(df)
  all_docs_df = pd.concat(temp)
  
  # counting number of sentences by label
  count = all_docs_df.groupby(['label'])['label'].count()
  
  # removing invalid labels
  labels_to_idx = data_loader.get_labels_to_idx()
  invalid_labels = [l for l, idx in labels_to_idx.items() if idx < 0]
  if len(invalid_labels) > 0:
    count = count.drop(labels=invalid_labels, errors='ignore')
  
  # replacing labels with acronyms
  count = count.rename(labels_to_acronym)
  
  return count

def get_count_df(data_loader):
  dic_docs_train, dic_docs_dev, dic_docs_test = data_loader.get_data()
  
  count_train = count(dic_docs_train, data_loader)
  count_dev = count(dic_docs_dev, data_loader)
  count_test = count(dic_docs_test, data_loader)
  
  return pd.DataFrame({'train': count_train, 'development': count_dev, 'test': count_test})

def plot_single(title, ax, count_df):
  ax.set_title(title)
  count_df.plot.barh(ax=ax, legend=0)
  ax.set_ylabel(None)
  #ax.set_xscale('log')

def main():
    count_df_original = get_count_df(data_manager.get_data_manager('original'))
    count_df_malik = get_count_df(data_manager.get_data_manager('7_roles'))
    count_df_4_labels = get_count_df(data_manager.get_data_manager('4_roles'))
    
    # ploting
    fig = plt.figure(figsize=(15, 4))
    
    ax_original = fig.add_subplot(131)
    plot_single('Original', ax_original, count_df_original)
    
    ax_7_labels = fig.add_subplot(132)
    plot_single('7-roles', ax_7_labels, count_df_malik)
    
    ax_4_labels = fig.add_subplot(133)
    plot_single('4-roles', ax_4_labels, count_df_4_labels)
    
    handles, labels_legend = ax_4_labels.get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05))
    
    plt.savefig('n_labels_per_dataset.pdf', bbox_inches='tight')

if __name__ == '__main__':
    main()
