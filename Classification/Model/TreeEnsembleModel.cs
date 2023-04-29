using System.Collections.Generic;
using System.IO;
using Classification.Model.DecisionTree;
using Math;

namespace Classification.Model
{
    public class TreeEnsembleModel : Model
    {
        private readonly List<DecisionTree.DecisionTree> _forest;

        /**
         * <summary> A constructor which sets the {@link ArrayList} of {@link DecisionTree} with given input.</summary>
         *
         * <param name="forest">An {@link ArrayList} of {@link DecisionTree}.</param>
         */
        public TreeEnsembleModel(List<DecisionTree.DecisionTree> forest)
        {
            _forest = forest;
        }

        public TreeEnsembleModel(string fileName)
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