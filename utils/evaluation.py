from sklearn.metrics import precision_score, recall_score, accuracy_score

def evaluate_model(train_embeddings, train_labels, test_embeddings, test_labels):
    # Predict for test set
    pred_labels = nearest_neighbor_predict(train_embeddings, test_embeddings)
    
    # Precision and Recall
    precision = precision_score(test_labels, pred_labels, average='weighted')
    recall = recall_score(test_labels, pred_labels, average='weighted')
    
    # Accuracy
    accuracy = accuracy_score(test_labels, pred_labels)
    
    return precision, recall, accuracy

def nearest_neighbor_predict(train_embeddings, test_embeddings):
    # Using Nearest Neighbor for simplicity
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(train_embeddings)
    _, indices = nn.kneighbors(test_embeddings)
    
    return indices
