using System;
using System.Collections.Generic;
using Classification.Model;
using Classification.Model.DecisionTree;
using Classification.Parameter;

namespace Classification.Classifier
{
    public class Bagging : Classifier
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
            var partition = trainSet.StratifiedPartition(0.2, new Random(parameters.GetSeed()));
            var forestSize = ((BaggingParameter) parameters).GetEnsembleSize();
            var forest = new List<DecisionTree>();
            for (var i = 0; i < forestSize; i++)
            {
                var bootstrapTrain = partition.Get(1).Bootstrap(i);
                var bootstrapPrune = partition.Get(0).Bootstrap(i);
                var tree =
                    new DecisionTree(new DecisionNode(new InstanceList.InstanceList(bootstrapTrain.GetSample()), null, null, false));
                tree.Prune(new InstanceList.InstanceList(bootstrapPrune.GetSample()));
                forest.Add(tree);
            }

            model = new TreeEnsembleModel(forest);
        }
    }
}