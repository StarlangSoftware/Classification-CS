using System.Collections.Generic;
using Classification.Model.DecisionTree;
using Classification.Parameter;

namespace Classification.Model
{
    public class RandomForestModel : TreeEnsembleModel
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
            _forest = new List<DecisionTree.DecisionTree>();
            for (var i = 0; i < forestSize; i++)
            {
                var bootstrap = trainSet.Bootstrap(i);
                _forest.Add(new DecisionTree.DecisionTree(new DecisionNode(new InstanceList.InstanceList(bootstrap.GetSample()), null,
                    (RandomForestParameter) parameters, false)));
            }

        }

        /// <summary>
        /// Loads the random forest model from an input file.
        /// </summary>
        /// <param name="fileName">File name of the random forest model.</param>
        public override void LoadModel(string fileName)
        {
            Load(fileName);
        }

    }
}