### visualize OP_sin in 3D

from importlib import reload
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import data_utils as du

for variant in ["sparse","dense"]:
    plt.close()
    df = du.load_dataset(dataset="OP_sin" if variant == "sparse" else "OP_sin_more", return_type="dataframe",select_features="all",test_split="none")
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract coordinates
    x = df['V (p. u.)']
    y = df['P (p. u.)']
    z = df['Q (p. u.)']
    
    colors = ['red' if val == 1 else 'green' for val in df['S']] 
    
    # Create scatter plot
    scatter_plot = ax.scatter(x, y, z, c=colors, s=2, alpha=1)
    
    # Set labels and title
    ax.set_xlabel('V (p. u.)')
    ax.set_ylabel('P (p. u.)')
    ax.set_zlabel('Q (p. u.)')
    #plt.title("3D Point Cloud")
    
    plt.savefig(f"plots/op_sin_pointcloud_{variant}_pu.png")
    
### visualize OP_sin as 2D slices

import pandas as pd
from plotnine import *

pd.set_option('display.precision',10)

df = du.load_dataset(dataset="OP_sin", return_type="dataframe")

for feat in ["V"]:
    print(feat,[v for v in df[feat].unique()])

print(df["V"])

#df["V_str"] = df["V"].apply(lambda x: str(x))

plot = (ggplot(df, aes(x="P",y="Q", color="factor(S)"))
+ scale_color_manual(values={1: "lightcoral", 0: "lightgreen"}, labels=["stable","unstable"], name="Ground truth")
+ geom_point(size=0.5)
+ facet_wrap("V", ncol=7, labeller="label_both")
+ theme_light()
+ theme(figure_size=(15,9), aspect_ratio=1))

plot.save("plots/OP_sin_slices.png")