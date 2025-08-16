import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pingouin as pg
#import ptitprince as pt
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('--csv_path', type=str, help='path to .CSV file')
parser.add_argument('--out_dir', type=str, help='path to write ouput files')
args = parser.parse_args()

DATA_DIR = args.csv_path
OUT_DIR = args.out_dir

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_DIR, sep='|')
df = df.drop(columns=['transcript', 'model_folder', 'meta_csv_path', 'absolute_audio_path', 'language_folder'])
# Replace any architecture starting with 'facebook/mms-tts-' to just 'facebook/mms-tts'
df['architecture'] = df['architecture'].str.replace(r'^facebook/mms-tts-\w+$', 'facebook/mms-tts', regex=True)
df['speaker']= df['original_file'].str.split('/').str[3]

df_shuffled = df.sample(frac=1).reset_index(drop=True)
df_shuffled.head()

def plot_raincloud(data, architecture, speaker=None):
    df = data[data['architecture'] == architecture]
    original_langs = (df.groupby('language')['is_original_language'].any())

    if speaker is not None:
        df = df[df['speaker'] == speaker]

    f, ax = plt.subplots(figsize=(12, 6))

    ax=pt.half_violinplot(x = 'language',
                          y = 'score',
                          data = df,                          
                          palette = 'tab10',
                          bw = .2,
                          cut = 0.,
                          scale = "count",
                          width = .6,
                          inner = None)
    
    ax=sns.stripplot(x = 'language',
                     y = 'score',
                     data = df,
                     palette = 'tab10',
                     size = 5,
                     jitter = 1,
                     alpha = .3)
    
    ax.set_title("Score distributions per language", fontsize=15)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=14)
    ax.set_xlabel("Language", fontsize=15)
    ax.set_ylabel("Score", fontsize=15)

    plt.yticks(fontsize=14)

    plt.tight_layout()
    out_file = os.path.join(OUT_DIR, 'raincloud.pdf')
    plt.savefig(out_file, format='pdf', bbox_inches='tight')

def run_tests(data, architecture):
    print(f'Analysis of {architecture} scores\n---------------------------------------\n')

    df = data[data['architecture'] == architecture]

    original_langs = (df.groupby('language')['is_original_language'].any())
    lang_order = sorted(df['language'].unique())

    plt.figure(figsize=(12, 6))
    ax = plt.gca()

    # Violin plots
    sns.violinplot(x='language',
                y='score',
                data=df,
                order=lang_order,
                palette='tab10',
                inner=None,
                fill=False,
                linewidth=.9,
                cut=0,
                ax=ax)  

    # Strip plots
    sns.stripplot(x='language',
                  y='score',
                  data=df,
                  order=lang_order,
                  palette='tab10',
                  alpha=.4,
                  size=4,
                  ax=ax)

    plt.title(f"Scores by Language\n{architecture}", fontsize=15)
    plt.xlabel("Language", fontsize=15)
    plt.ylabel("Score", fontsize=15)
    plt.xticks(fontsize=13)
    
    tick_labels = ax.get_xticklabels()
    for tick_label in tick_labels:
        lang = tick_label.get_text()
        if original_langs.get(lang, False):
            tick_label.set_color('red')  # or any other color
        else:
            tick_label.set_color('black')  # default

    plt.tight_layout()
    out_file = os.path.join(OUT_DIR, 'strip.pdf')
    plt.savefig(out_file, format='pdf', bbox_inches='tight')
    plt.show()

    # Pairwise tests

    pairwise_stats = pg.pairwise_tests(
        dv='score',
        between='language',
        alpha=0.01,
        data=df,
        padjust='bonf',  
        effsize='CLES',  
        parametric=False)  

    #print(pairwise_stats)  
    
    delta_matrix = pairwise_stats.pivot(index='A', columns='B', values='CLES')
    mask = np.tril(np.ones(delta_matrix.shape), k=-1).astype(bool)
    delta_matrix = delta_matrix.mask(mask)

    annot_matrix = pd.DataFrame(index=delta_matrix.index, columns=delta_matrix.columns)

    # Iterate through pairwise_stats
    for _, row in pairwise_stats.iterrows():
        a, b = row['A'], row['B']
        cles = row['CLES']
        pval = row['p-corr']
        
        if pd.notnull(cles):
            if pval < 0.001:
                star = '***'
            elif pval < 0.01:
                star = '**'
            elif pval < 0.05:
                star = '*'
            else:
                star = ""
            formatted = f"{cles:.2f}\n{star}"
            annot_matrix.loc[a, b] = formatted

    annot_matrix = annot_matrix.mask(mask)

    # Plot heatmap

    plt.figure(figsize=(10, 10))
    ax = sns.heatmap(delta_matrix,
                annot=annot_matrix,
                annot_kws={"size": 15, "weight": "bold", 'color': "black"},
                #cbar_kws={"label": "CLES"},
                cmap='coolwarm',
                fmt='',
                center=0.5,
                vmin=0,
                vmax=1,
                linewidths=0.5)
    ax.set_xlabel(f'Group B\n\nMann-Whitney U p-value: * p<0.05 | ** p<0.01 | *** p<0.001', fontsize=20)
    ax.set_ylabel(f'Group A', fontsize=20)

    cbar = ax.collections[0].colorbar
    cbar.set_label('CLES', size=20)

    ax.set_yticklabels(ax.get_yticklabels(), rotation=45, fontsize=15, weight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=15, weight='bold')
    ax.set_title("Pairwise tests between languages", fontsize=20)
    plt.tight_layout()
    out_file = os.path.join(OUT_DIR, 'cles.pdf')
    plt.savefig(out_file, format='pdf', bbox_inches='tight')
    plt.show()

    stats = df.groupby('language')['score'].agg(['mean', 'std']).reset_index()
    #print(stats)

run_tests(df, 'facebook/mms-tts')


