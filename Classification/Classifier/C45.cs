using System;
using Classification.Model.DecisionTree;
using Classification.Parameter;

namespace Classification.Classifier
{
    public class C45 : Classifier
    {
        /**
         * <summary> Training algorithm for C4.5 univariate decision tree classifier. 20 percent of the data are left aside for pruning
         * 80 percent of the data is used for constructing the tree.</summary>
         *
         * <param name="trainSet">  Training data given to the algorithm.</param>
         * <param name="parameters">-</param>
         */
        public override void Train(InstanceList.InstanceList trainSet, Parameter.Parameter parameters)
        {
            DecisionTree tree;
            if (((C45Parameter) parameters).IsPrune())
            {
                var partition = trainSet.StratifiedPartition(
                    ((C45Parameter) parameters).GetCrossValidationRatio(), new Random(parameters.GetSeed()));
                tree = new DecisionTree(new DecisionNode(partition.Get(1), null, null, false));
                tree.Prune(partition.Get(0));
            }
            else
            {
                tree = new DecisionTree(new DecisionNode(trainSet, null, null, false));
            }

            model = tree;
        }

        public override void LoadModel(string fileName)
        {
            model = new DecisionTree(fileName);
        }
    }
}