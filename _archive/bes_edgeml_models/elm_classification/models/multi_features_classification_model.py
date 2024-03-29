import argparse
from torch import nn
from bes_edgeml_models.base.models.multi_features_ds_v2_model import MultiFeaturesDsV2Model

class MultiFeaturesClassificationModel(MultiFeaturesDsV2Model):

    def __init__(self, args):
        super(MultiFeaturesClassificationModel, self).__init__(args)
        self.fc1 = nn.Linear(in_features=self.input_features, out_features=args.fc1_size)
        self.fc2 = nn.Linear(in_features=args.fc1_size, out_features=args.fc2_size)
        self.fc3 = nn.Linear(in_features=args.fc2_size, out_features=1)

    def forward(self, x):
        x = super(MultiFeaturesClassificationModel, self).forward(x)
        x = self.relu(self.dropout(self.fc1(x)))
        x = self.relu(self.dropout(self.fc2(x)))
        x = self.fc3(x)
        return x