import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from matplotlib import gridspec
from sklearn.inspection import PartialDependenceDisplay
from sklearn.inspection import permutation_importance
import sys
from joblib import dump
from atlasvras.utils.prettify import vra_colors, label_to_color

plt.style.use('vra')


##############################################
#    ____   ____ ______       __  __   ___ 
#   / __/  / __//_  __/      / / / /  / _ \
#  _\ \   / _/   / /        / /_/ /  / ___/
# /___/  /___/  /_/         \____/  /_/    
#
##############################################

######## UTILITY DICTS AND FUNCTIONS #######


dict_type_to_preal = {'garbage': 0,
                      'pm': 0,
                      'galactic': 1,
                      'good': 1
                     }

dict_type_to_pgal = {'garbage': 0,
                      'pm': 1,
                      'galactic': 1,
                      'good': 0
                     }



def plot_1Dscore(PRED, Y,label=None,
                 alert_type ='good', 
                 ax=None, color = vra_colors['blue'],
                 ):
    "A quick function to plot the 1D score distribution for a specific alert type"
    mask = Y.type == alert_type

    if ax is None:
        f, ax = plt.subplots(facecolor='black', figsize=(12, 8))
        ax.set_ylabel('Density')
        ax.set_xlabel('Score')


    ax.hist(PRED[mask],
            color=color, 
            edgecolor=color,
            bins=50, 
            alpha=0.7,
            density=True,
            label=f'{label}',
           )

##### GET CL PARAMETERS ########
# get the following constants from the command line using sys.argv 

RANDOM_STATE = 42
CLASS_WEIGHT = 'balanced'
try:
    real_or_gal = sys.argv[1]
    LR = float(sys.argv[2])
    L2 = float(sys.argv[3])
except IndexError:
    raise IndexError("You didn't provide enough arguments. Expected: real_or_gal, LR, L2")
##########################################




################################
#     __   ____    ___    ___ 
#   / /  / __ \  / _ |  / _ \
#  / /__/ /_/ / / __ | / // /
# /____/\____/ /_/ |_|/____/
#   THE DATA AND THE LABELS
###############################
RELATIVE_PATH = '../../data/features_and_labels_csv/update/'
X_test_unbalanced = pd.read_csv(f'{RELATIVE_PATH}X_test_unbalanced.csv', index_col=0)
y_test_unbalanced = pd.read_csv(f'{RELATIVE_PATH}y_test_unbalanced.csv', index_col=0)

X_train = pd.read_csv(f'{RELATIVE_PATH}X_train.csv', index_col = 0)
y_train = pd.read_csv(f'{RELATIVE_PATH}y_train.csv', index_col = 0)

# TURN THE MULTI-CLASS LABELS INTO BINARY LABELS
if real_or_gal == 'real':
    y_train_binary = y_train.map(lambda x: dict_type_to_preal[x]) 
    y_test_binary = y_test_unbalanced.map(lambda x: dict_type_to_preal[x])  
elif real_or_gal == 'gal':
    y_train_binary = y_train.map(lambda x: dict_type_to_pgal[x])
    y_test_binary = y_test_unbalanced.map(lambda x: dict_type_to_pgal[x])     


###########################             
#      ___   __  __   _  __
#    / _ \ / / / /  / |/ /
#   / , _// /_/ /  /    / 
#  /_/|_| \____/  /_/|_/ 
# THE TRAINING AND TESTING
###########################

scorer = HistGradientBoostingClassifier(learning_rate=LR,
                                             random_state=RANDOM_STATE,
                                             class_weight=CLASS_WEIGHT,
                                             l2_regularization=L2
                                            )

scorer.fit(X_train, y_train_binary)


pred_TRAIN  = scorer.predict_proba(X_train).T[1]
pred_TEST = scorer.predict_proba(X_test_unbalanced).T[1]

#################################
#    ___    __   ____  ______
#   / _ \  / /  / __ \/_  __/
#  / ___/ / /__/ /_/ / / /   
# /_/    /____/\____/ /_/    
#
#################################


#################################
# 1. SCORE DISTRIBUTONS BY TYPE
#################################

##### [0,0] GARBAGE
LABEL = 'garbage'
f, ax = plt.subplots(ncols = 2, nrows=2, facecolor='black', figsize=(12, 12))

plot_1Dscore(pred_TRAIN, y_train,
             label='Training', color='white',
             ax=ax[0,0],alert_type=LABEL)


plot_1Dscore(pred_TEST, y_test_unbalanced,
             label='Test', color=vra_colors['red'],
             ax=ax[0,0],alert_type=LABEL)


ax[0,0].set_ylabel('Density')
ax[0,0].set_title(f'{LABEL} [Real Score]',loc='right')
ax[0,0].legend()


##### [0,1] PROPER MOTION STARS
LABEL = 'pm' 
plot_1Dscore(pred_TRAIN, y_train,
             label='Training', color='white',
             ax=ax[0,1],alert_type=LABEL)


plot_1Dscore(pred_TEST, y_test_unbalanced,
             label='Test', color=vra_colors['red'],
             ax=ax[0,1],alert_type=LABEL)


ax[0,1].set_ylabel('Density')
ax[0,1].set_title(f'{LABEL} [Gal Score]',loc='right')
ax[0,1].legend()

##### [1,0] GALACTIC
LABEL = 'galactic'

plot_1Dscore(pred_TRAIN, y_train,
             label='Training', color='white',
             ax=ax[1,0],alert_type=LABEL)


plot_1Dscore(pred_TEST, y_test_unbalanced,
             label='Test', color=vra_colors['red'],
             ax=ax[1,0],alert_type=LABEL)


ax[1,0].set_ylabel('Density')
ax[1,0].set_title(f'{LABEL} [Real Score]',loc='right')
ax[1,0].legend()


##### [1,1] GOOD
LABEL = 'good'  
plot_1Dscore(pred_TRAIN, y_train,
             label='Training', color='white',
             ax=ax[1,1],alert_type=LABEL)


plot_1Dscore(pred_TEST, y_test_unbalanced,
             label='Test', color=vra_colors['red'],
             ax=ax[1,1],alert_type=LABEL)


ax[1,1].set_ylabel('Density')
ax[1,1].set_title(f'{LABEL} [Gal Score]',loc='right')
ax[1,1].legend()

plt.suptitle(f'{real_or_gal}\nLR:{LR} | L2:{L2}\nrandomsplit', fontsize=26)
plt.savefig(f'hp_tuning_update/figures/{real_or_gal}_1Dscore_LR{LR}_L2{L2}_randomsplit.png',
        bbox_inches='tight') # FIX THIS WITH THE PROPER HP SEQUENCE SEE CONSTANTS ABOVE



#################################
# 2. PERMUTATION IMPORTANCE
#################################

permimp = permutation_importance(scorer, 
                                      X_train, 
                                      y_train_binary, 
                                      n_repeats=16, 
                                      random_state=42, 
                                      n_jobs=5
                                     )



sorted_importances_idx = permimp.importances_mean.argsort()
importances = pd.DataFrame(
    permimp.importances[sorted_importances_idx].T,
    columns=X_train.columns[sorted_importances_idx],
)
ax = importances.plot.box(vert=False, whis=10)
ax.set_title(f"Permutation Importances {real_or_gal}")
ax.axvline(x=0, color="k", linestyle="--")
ax.set_xlabel("Decrease in accuracy score")
ax.figure.tight_layout()
plt.title(f'{real_or_gal}_LR{LR}_L2{L2}_randomsplit')
## SAVE THE PLOT
plt.savefig(f'hp_tuning_update/figures/{real_or_gal}_permimp_real_LR{LR}_L2{L2}_randomsplit.png')

##################
#      __                           __    __                 
#  ___/ / __ __  __ _    ___       / /_  / /  ___            
# / _  / / // / /  ' \  / _ \     / __/ / _ \/ -_)           
# \_,_/  \_,_/ /_/_/_/ / .__/     \__/ /_//_____/     __     
#                     /_/    __ _  ___  ___/ / ___   / /  ___
#                           /  ' \/ _ \/ _  / / -_) / /  (_-<    #
#                          /_/_/_/\___/\_,_/  \__/ /_/  /___/    #
#                                                                #
#                                            #####################

print('dumping model')
dump(scorer, f'hp_tuning_update/models/{real_or_gal}_scorer_LR{LR}_L2{L2}_randomsplit.joblib')
