using System.Collections.Generic;
using System.IO;
using Classification.Model.DecisionTree;
using Math;

namespace Classification.Model
{
    public abstract class TreeEnsembleModel : Model
    {
        protected List<DecisionTree.DecisionTree> _forest;
        
        /// <summary>
        /// Loads a tree ensemble model such as Random Forest model or Bagging model from an input model file.
        /// </summary>
        /// <param name="fileName">Model file name.</param>
        public void Load(string fileName)
        {
            var input = new StreamReader(fileName);
            var numberOfTrees = int.Parse(input.ReadLine());
            _forest = new List<DecisionTree.DecisionTree>();
            for (var i = 0; i < numberOfTrees; i++){
                _forest.Add(new DecisionTree.DecisionTree(new DecisionNode(input)));
            }
            input.Close();
        }
        
        /**
         * <summary> The predict method takes an {@link Instance} as an input and loops through the {@link ArrayList} of {@link DecisionTree}s.
         * Makes prediction for the items of that ArrayList and returns the maximum item of that ArrayList.</summary>
         *
         * <param name="instance">Instance to make prediction.</param>
         * <returns>The maximum prediction of a given Instance.</returns>
         */
        public override string Predict(Instance.Instance instance)
        {
            var distribution = new DiscreteDistribution();
            foreach (var tree in _forest)
            {
                var predictedLabel = tree.Predict(instance);
                if (predictedLabel != null)
                {
                    distribution.AddItem(predictedLabel);
                }
            }
            return distribution.GetMaxItem();
        }

        /// <summary>
        /// Calculates the posterior probability distribution for the given instance according to ensemble tree model.
        /// </summary>
        /// <param name="instance">Instance for which posterior probability distribution is calculated.</param>
        /// <returns>Posterior probability distribution for the given instance.</returns>
        public override Dictionary<string, double> PredictProbability(Instance.Instance instance)
        {
            var distribution = new DiscreteDistribution();
            foreach (var tree in _forest) {
                distribution.AddItem(tree.Predict(instance));
            }
            return distribution.GetProbabilityDistribution();
        }
    }
}