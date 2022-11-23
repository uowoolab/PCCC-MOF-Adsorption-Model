import torch
import warnings
import sys
import os.path
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from calc_descriptors import calc_geo_props, calc_rdfs
import pickle
import argparse

# Define globally, needed by Net3 class and main function
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')


# Class for 3-layer models
class Net3(nn.Module):
    
    def __init__(self):
        super(Net3, self).__init__()

    #forward function applies activation function
    #to input and sends it to output (x is input tensor)
    def forward(self, x):
        if device == 'cuda:0':
            x.cuda(device)
        x = self.dropout(F.relu(self.hidden1(x)))
        x = self.dropout(F.relu(self.hidden2(x)))
        x = self.dropout(F.relu(self.hidden3(x)))
        x = F.relu(self.output(x))
        return x

def main(cif, zeo_exe, discard_geo):
    start_time = datetime.now()
    print("\nStart: ", start_time.strftime("%c"))

    print("Predicting adsorption properties for: {}".format(cif))

    print("Device for prediction: ", device)

    print("\n\tComputing geometric descriptors...")
    geo_props = calc_geo_props(cif,
                               zeo_exe=zeo_exe,
                               discard_geo=discard_geo)
    print("\n\tComputing RDFs...")
    rdfs = calc_rdfs(name=cif,
                     props=["electronegativity",
                            "vdWaalsVolume",
                            "polarizability"],
                     smooth=-10,
                     factor=0.001)

    features = rdfs + geo_props

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    f.close()

    features = scaler.transform(np.array([features]))
    features = torch.FloatTensor(features)
    features = features.to(device)

    print("\n\tLoading in PyTorch models...")
    wc_model = torch.load("wc_model.pt", map_location=torch.device(device))
    wc_model.eval()
    sel_model = torch.load("sel_model.pt", map_location=torch.device(device))
    sel_model.eval()

    print("\n\tMaking predictions on the dataset...")
    y_predict_wc = wc_model(features)
    y_predict_sel = sel_model(features)
    y_predict_wc = y_predict_wc.to('cpu').detach().numpy()
    y_predict_sel = y_predict_sel.to('cpu').detach().numpy()

    print("\nPredicted working capacity (mmol/g): {}".format(np.round(
                                                           float(y_predict_wc[0][0]),
                                                           2)))
    print("Predicted CO2/N2 selectivity: {}\n".format(np.round(
                                                    float(y_predict_sel[0][0]),
                                                    2)))
    print("Successful termination.")
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print("End: ", end_time.strftime("%c"))
    print("Total time: {0:.1f} s\n".format(elapsed_time.total_seconds()))

if __name__ == "__main__":

    ap = argparse.ArgumentParser(
         description = "Predict working capacity for a MOF under " + 
                        "post-combustion carbon capture conditions")
    ap.add_argument("cif", type=str,
                    help="CIF file of the structure")
    ap.add_argument("zeo_exe", type=str,
                    help="Path of the zeo++ executable")
    ap.add_argument("-discard_geo_props", type=str, default=True,
                    help="Whether to discard the geometric properties." + 
                          " If false, they are written in the present directory.")
    args = ap.parse_args()
    filename = args.cif
    zeo_exe = args.zeo_exe
    discard_geo = args.discard_geo_props
    main(cif=filename,
         zeo_exe=zeo_exe,
         discard_geo=discard_geo)

