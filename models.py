import os
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

    
# https://github.com/adam-katona/dota2_death_prediction/blob/3c6e86e7aa64deaad7bca2f5951428fb6bb50719/models.py
class SimpleFF(torch.nn.Module):
    def __init__(self,num_features,num_labels):
        super(SimpleFF, self).__init__()
        
        self.linear1 = torch.nn.Linear(in_features=num_features,out_features=200)
        self.linear2 = torch.nn.Linear(in_features=200,out_features=100)
        self.linear3 = torch.nn.Linear(in_features=100,out_features=num_labels)

    def forward(self, features):
        x = features
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class EmbeddingFF(torch.nn.Module):
    def __init__(self, num_features, num_labels, categorical_features_info={}, config={}):
        """
        This model uses embeddings for more complex categorical features, and adds dropout
        Args:
            num_features (int): Number of continuous/numerical features
            num_labels (int): Number of output labels
            categorical_features_info (list): List of Dictionary mapping categorical feature names 
                                              to their number of unique categories
                                              Example: [{'feature_names': ['city'], 'num_categories': 10, 'embedding_dim': 3}]
            config (dict): Dictionary of tunable parameters: "dropout", 
        """
        super(EmbeddingFF, self).__init__()
        
        self.categorical_embeddings = nn.ModuleDict()
        total_embedding_dim = 0
        
        self.assigned_features = set()
        
        for group in categorical_features_info:
            feature_names = group['feature_names']
            num_categories = group['num_categories']
            
            embedding_dim = group.get(
                'embedding_dim', 
                min(50, (num_categories + 1) // 2)
            )
            
            shared_embedding = nn.Embedding(
                num_embeddings=num_categories, 
                embedding_dim=embedding_dim
            )

            print(f"embedding dim {embedding_dim} for:")
            
            for feature in feature_names:
                print(feature)
                self.categorical_embeddings[feature] = shared_embedding
                self.assigned_features.add(feature)
                total_embedding_dim += embedding_dim
        
        total_input_features = num_features + total_embedding_dim

        print(f"total input features {total_input_features}")
        
        self.linear1 = torch.nn.Linear(in_features=total_input_features, out_features=200)
        self.linear2 = torch.nn.Linear(in_features=200, out_features=100)
        self.linear3 = torch.nn.Linear(in_features=100, out_features=num_labels)
        
        self.dropout = nn.Dropout(config.get("dropout", 0.2))

    def forward(self, numerical_features, categorical_features):
        """
        Args:
            numerical_features (torch.Tensor): Tensor of numerical features
            categorical_features (dict): Dictionary of categorical feature tensors
                                         Example: {'gender': gender_tensor, 'city': city_tensor}
        """
        for feature in categorical_features:
            if feature not in self.assigned_features:
                raise ValueError(f"Embedding not configured for feature: {feature}")
        
        categorical_embeddings = []
        for feature_name, feature_tensor in categorical_features.items():
            embedded_feature = self.categorical_embeddings[feature_name](feature_tensor)
            # print(f"{feature_name} - Original tensor shape: {feature_tensor.shape}")
            # print(f"{feature_name} - Embedded feature shape: {embedded_feature.shape}")
            categorical_embeddings.append(embedded_feature.view(embedded_feature.size(0), -1))
        
        categorical_embeddings = torch.cat(categorical_embeddings, dim=1)
        # print("Categorical embeddings total shape:", categorical_embeddings.shape)

        # for emb in categorical_embeddings:
        #     print("Flattened embedding shape:", emb.shape)
            
        x = torch.cat([numerical_features, categorical_embeddings], dim=1)

        # print("Input tensor shape:", x.shape)
        # print("Expected input features for linear1:", self.linear1.in_features)
        
        x = torch.relu(self.linear1(x))
        x = self.dropout(x)
        x = torch.relu(self.linear2(x))
        x = self.dropout(x)
        x = self.linear3(x)
        
        return x