using System.Collections.Generic;
using Classification.Model;
using Classification.Model.DecisionTree;
using Classification.Parameter;

namespace Classification.Classifier
{
    public class RandomForest : Classifier
    {
        /**
         * <summary> Training algorithm for random forest classifier. Basically the algorithm creates K distinct decision trees from
         * K bootstrap samples of the original training set.</summary>
         *
         * <param name="trainSet">  Training data given to the algorithm</param>
         * <param name="parameters">Parameters of the bagging trees algorithm. ensembleSize returns the number of trees in the random forest.</param>
         */
        public override void Train(InstanceList.InstanceList trainSet, Parameter.Parameter parameters)
        {
            var forestSize = ((RandomForestParameter) parameters).GetEnsembleSize();
            var forest = new List<DecisionTree>();
            for (var i = 0; i < forestSize; i++)
            {
                var bootstrap = trainSet.Bootstrap(i);
                forest.Add(new DecisionTree(new DecisionNode(new InstanceList.InstanceList(bootstrap.GetSample()), null,
                    (RandomForestParameter) parameters, false)));
            }

            model = new TreeEnsembleModel(forest);
        }
    }
}