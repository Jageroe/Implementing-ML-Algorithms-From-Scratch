""" 
I got the basic idea from here: https://insidelearningmachines.com/build-a-decision-tree-in-python/
But since I didn't like a lot of things from the original code,
I made my own version by changing several parts of the code, and using my own naming and doc conventions. 
"""

# imports
from __future__ import annotations
from typing import Tuple
from abc import ABC,abstractmethod
from scipy import stats
import numpy as np


class Node(object):
    """
    Class to define & control tree nodes
    """
    
    def __init__(self) -> None:
        """
        Initializer for a Node class instance
        """
        self._split = None
        self._feature = None
        self._left = None
        self._right = None
        self.leaf_value = None

    def set_params(self, split: float, feature: int) -> None:
        """
        Set the split and feature parameters for this node.
        
        Args:
            split: value to split feature on.
            feature: index of feature to be used in splitting.
        """
        self._split   = split
        self._feature = feature
        
    def get_params(self) -> Tuple[float,int]:
        """
        Get the split and feature parameters for this node.
        
        Returns:
            Tuple containing (split, feature) pair.
        """
        return(self._split, self._feature)    
        
    def set_children(self, left: Node, right: Node) -> None:
        """
        Set the left/right child nodes for the current node.
        
        Args:
            left: left child node.
            right: right child node.
        """
        self._left  = left
        self._right = right
    
    def get_left_node(self) -> Node:
        """
        Get the left child node.
        
        Returns:
            left child node.
        """
        return(self._left)
    
    def get_right_node(self) -> Node:
        """
        Get the right child node
        
        Returns:
            right child node
        """
        return(self._right)
    

class DecisionTree(ABC):
    """
    Base class to encompass the CART algorithm
    """
    
    def __init__(self, 
                 max_depth: int=None, 
                 min_samples_split: int=2) -> None:
        """
        Initialize a DecisionTree instance.
        
        Args:
            max_depth: maximum depth the tree can grow. Defaults to None.
            min_samples_split: minimum number of samples required to split a node. Defaults to 2.
        """

        self.root = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    @abstractmethod
    def _impurity(self, data: np.array) -> None:
        """
        Abstract function to define the impurity
        """
        pass
        
    @abstractmethod
    def _leaf_value(self, data: np.array) -> None:
        """
        Abstract function to compute the value at a leaf node
        """
        pass

    def _grow(self, 
              node: Node, 
              data: np.array, 
              level: int) -> None:
        
        """
        Recursive function to grow the tree during training.
        
        Args:
            node: input tree node.
            data: sample of data at node.
            level: depth level in the tree for node.
        """

        # True if the the current level hasn't reached the max depth level
        under_max_depth = (self.max_depth is None) or (self.max_depth >= (level+1))
        # True if We have equal or more samples than the min_samples_split
        has_enough_sample = (self.min_samples_split <= data.shape[0])
        # True if we have multiple type of classes in the target feature
        multiple_cls = np.unique(data[:,-1]).shape[0] != 1
        
        # if these conditions are true, we can grow another node
        if under_max_depth and has_enough_sample and multiple_cls:
        
            # I will try every combination of features and split value, 
            # and I will store the best case the beloving variables
            best_ip = None
            best_feat = None
            best_split = None
            best_data_l = None
            best_data_r = None
            
            # Iterate through the possible feature/split combinations
            for curr_feat in range(data.shape[1]-1):  # all possible feature
                for curr_val in np.unique(data[:,curr_feat]):  # all unique data value
                    
                    # for the current feature/split combination, split the dataset
                    curr_data_l = data[data[:,curr_feat]<=curr_val]
                    curr_data_r = data[data[:,curr_feat]>curr_val]

                    # ensure we have non-empty arrays
                    if curr_data_l.size and curr_data_r.size:

                        # calculate the impurity
                        curr_ip_l = (curr_data_l.shape[0]/data.shape[0])*self._impurity(curr_data_l)
                        curr_ip_r = (curr_data_r.shape[0]/data.shape[0])*self._impurity(curr_data_r)
                        curr_ip  = curr_ip_l + curr_ip_r
                        
                        # update parameters if the current ip is lower than the lowest so far
                        if (best_ip is None) or (curr_ip < best_ip):
                            best_ip = curr_ip
                            best_feat = curr_feat
                            best_split = curr_val
                            best_data_l = curr_data_l
                            best_data_r = curr_data_r
                            
            # set the current node's parameters
            node.set_params(best_split,best_feat)

            # Declare child nodes
            left_node = Node()
            right_node = Node()
            node.set_children(left_node,right_node)

            # Grow the further child nodes"
            self._grow(node.get_left_node(),best_data_l,level+1)
            self._grow(node.get_right_node(),best_data_r,level+1)
                        
        # is a leaf node
        else:
            # set the node value & return
            node.leaf_value = self._leaf_value(data)
            return
     
    def _traverse(self, node: Node, x: np.array) -> int | float:
        """
        Recursive function to traverse the (trained) tree.
        
        Args:
            node: current node in the tree.
            x: data sample being considered.
        
        Returns:
            leaf value corresponding to x.
        """         
        # check if we're in a leaf node?
        if node.leaf_value is None:
            # get parameters at the node
            (split,feat) = node.get_params()
            # decide to go left or right?
            if (x[feat] <= split):
                return(self._traverse(node.get_left_node(),x))
            else:
                return(self._traverse(node.get_right_node(),x))
        else:
            # return the leaf value
            return(node.leaf_value)
    
    def train(self, X: np.array, y: np.array) -> None:
        """
        Trains the CART model
        
        Args:
            X: input set of predictor features
            y: input set of labels
        """        
        # prepare the input data
        data = np.concatenate((X,y.reshape(-1,1)),axis=1)
        # set the root node of the tree
        self.root = Node()
        # build the tree
        self._grow(self.root,data,1)
        
    def predict(self, X: np.array) -> np.array:
        """
        Make predictions from the trained CART model
        
        Args:
            X: input set of predictor features
        Returns:
            array of predictied values
        """
        # iterate through the rows of Xin
        preds = []
        for r in range(X.shape[0]):
            preds.append(self._traverse(self.root,X[r,:]))
        # return predictions
        return(np.array(preds).flatten())
    

class DecisionTreeClassifier(DecisionTree):
    """
    Decision Tree Classifier
    """
    
    def __init__(
        self, max_depth: int=None, 
        min_samples_split: int=2, 
        loss: str='gini') -> None:
        
        """
        Initializer
        
        Args:
            max_depth: maximum depth the tree can grow
            min_samples_split: minimum number of samples required to split a node
            loss: loss function to use during training
        """
        super().__init__(max_depth,min_samples_split)
        self.loss = loss   
    
    def _gini(self, data: np.array) -> float:
        """
        Function to define the gini impurity
        
        Args:
            data: data to compute the gini impurity over
        Returns:
            Gini impurity for data
        """        
        # initialize the output
        g_imp = 0
        # iterate through the unique classes
        for u_cls in np.unique(data[:,-1]):
            # compute p for the current u_cls
            cls_prop = data[data[:,-1]==u_cls].shape[0]/data.shape[0]
            # compute term for the current u_cls
            g_imp += cls_prop*(1-cls_prop)
        # return gini impurity
        return(g_imp)
    
   
    def _entropy(self, data: np.array) -> float:
        """
        Function to define the Shannon entropy
        
        Args:
            data: data over which to compute the Shannon entropy.
        Returns:
            Shannon entropy for the data.
        """        
        # Initialize the entropy
        entropy = 0
        # Iterate through the unique classes
        for u_cls in np.unique(data[:, -1]):
            # Compute proportion of current class
            cls_prop = data[data[:, -1] == u_cls].shape[0] / data.shape[0]
            # Compute contribution of the current class to the entropy
            entropy -= cls_prop * np.log2(cls_prop)
        # Return entropy
        return entropy
    
    def _impurity(self, data: np.array) -> float:
        """
        Protected function to define the impurity
        
        Args:
            data: data to compute the impurity metric over
        Output:
            impurity metric for data        
        """        
        # use the selected loss function to calculate the node impurity
        ip = None
        if self.loss == 'gini':
            ip = self._gini(data)
        elif self.loss == 'entropy':
            ip = self._entropy(data)
        # return results
        return(ip)
    
    def _leaf_value(self, data: np.array) -> int:
        """
        Protected function to compute the value at a leaf node
        
        Args:
            data: data to compute the leaf value
        Output:
            mode of data         
        """        
        return(stats.mode(data[:,-1],keepdims=False)[0])
    


class DecisionTreeRegressor(DecisionTree):
    """
    Decision Tree Regressor
    """
    
    def __init__(self, max_depth: int = None, min_samples_split: int = 2, loss: str = 'mse') -> None:
        """
        Initializer

        Args:
            max_depth: maximum depth the tree can grow.
            min_samples_split: minimum number of samples required to split a node.
            loss: loss function to use during training. Supports 'mse' for Mean Squared Error
                  and 'mae' for Mean Absolute Error.
        """
        super().__init__(max_depth, min_samples_split)
        self.loss = loss  
    
    def _mse(self, data: np.array) -> float:
        """
        Function to define the Mean Squared Error (MSE).

        Args:
            data: data to compute the MSE over.

        Returns:
            Mean Squared Error for the data.
        """
        y_mean = np.mean(data[:, -1])
        mse = np.mean((data[:, -1] - y_mean) ** 2)
        return mse

    
    def _mae(self, data: np.array) -> float:
        """
        Function to define the Mean Absolute Error (MAE).

        Args:
            data: data to compute the MAE over.

        Returns:
            Mean Absolute Error for the data.
        """
        y_mean = np.mean(data[:, -1])
        mae = np.mean(np.abs(data[:, -1] - y_mean))
        return mae

    
    def _impurity(self, data: np.array) -> float:
        """
        Function to define the impurity
        
        Args:
            data: data to compute the impurity metric over.

        Returns:
            Impurity metric for the data.
        """            
        # use the selected loss function to calculate the node impurity
        ip = None
        if self.loss == 'mse':
            ip = self._mse(data)
        elif self.loss == 'mae':
            ip = self._mae(data)
        # return results
        return(ip)
    
    def _leaf_value(self, data: np.array) -> float:
        """
        Function to compute the value at a leaf node, based on the target values of the data reaching the leaf.

        Args:
            data: data to compute the leaf value from.

        Returns:
            The mean of the target values in the data.
        """
        return np.mean(data[:, -1])