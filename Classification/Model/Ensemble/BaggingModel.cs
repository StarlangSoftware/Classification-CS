using System.Collections.Generic;
using Classification.Model.DecisionTree;
using Classification.Parameter;

namespace Classification.Model
{
    public class BaggingModel : TreeEnsembleModel
    {
        /**
         * <summary> Bagging bootstrap ensemble method that creates individuals for its ensemble by training each classifier on a random
         * redistribution of the training set.
         * This training method is for a bagged decision tree classifier. 20 percent of the instances are left aside for pruning of the trees
         * 80 percent of the instances are used for training the trees. The number of trees (forestSize) is a parameter, and basically
         * the method will learn an ensemble of trees as a model.</summary>
         *
         * <param name="trainSet">  Training data given to the algorithm.</param>
         * <param name="parameters">Parameters of the bagging trees algorithm. ensembleSize returns the number of trees in the bagged forest.</param>
         */
        public override void Train(InstanceList.InstanceList trainSet, Parameter.Parameter parameters)
        {
            var forestSize = ((BaggingParameter) parameters).GetEnsembleSize();
            _forest = new List<DecisionTree.DecisionTree>();
            for (var i = 0; i < forestSize; i++)
            {
                var bootstrap = trainSet.Bootstrap(i);
                var tree = new DecisionTree.DecisionTree(new DecisionNode(new InstanceList.InstanceList(bootstrap.GetSample()), null, null, false));
                _forest.Add(tree);
            }
        }

        /// <summary>
        /// Loads the Bagging ensemble model from an input file.
        /// </summary>
        /// <param name="fileName">File name of the decision tree model.</param>
        public override void LoadModel(string fileName)
        {
            Load(fileName);
        }

    }
}