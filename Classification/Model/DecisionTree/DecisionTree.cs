using System;
using System.Collections.Generic;
using System.IO;
using Classification.Instance;
using Classification.Parameter;

namespace Classification.Model.DecisionTree
{
    public class DecisionTree : ValidatedModel
    {
        protected DecisionNode Root;

        public DecisionTree()
        {
        }

        /**
         * <summary> Constructor that sets root node of the decision tree.</summary>
         *
         * <param name="root">DecisionNode type input.</param>
         */
        public DecisionTree(DecisionNode root)
        {
            Root = root;
        }

        /// <summary>
        /// Loads a decision tree model from an input model file.
        /// </summary>
        /// <param name="fileName">Model file name.</param>
        protected void Load(string fileName)
        {
            var input = new StreamReader(fileName);
            Root = new DecisionNode(input);
            input.Close();
        }
        
        /// <summary>
        /// Loads a decision tree model from an input model file.
        /// </summary>
        /// <param name="fileName">Model file name.</param>
        public DecisionTree(string fileName)
        {
            Load(fileName);
        }

        /**
         * <summary> The predict method  performs prediction on the root node of given instance, and if it is null, it returns the possible class labels.
         * Otherwise it returns the returned class labels.</summary>
         *
         * <param name="instance">Instance make prediction.</param>
         * <returns>Possible class labels.</returns>
         */
        public override string Predict(Instance.Instance instance)
        {
            var predictedClass = Root.Predict(instance);
            if (predictedClass == null && instance is CompositeInstance)
            {
                predictedClass = ((CompositeInstance)instance).GetPossibleClassLabels()[0];
            }

            return predictedClass;
        }

        /// <summary>
        /// Calculates the posterior probability distribution for the given instance according to Decision tree model.
        /// </summary>
        /// <param name="instance">Instance for which posterior probability distribution is calculated.</param>
        /// <returns>Posterior probability distribution for the given instance.</returns>
        public override Dictionary<string, double> PredictProbability(Instance.Instance instance)
        {
            return Root.PredictProbabilityDistribution(instance);
        }

        /**
         * <summary> The prune method takes an {@link InstanceList} and  performs pruning to the root node.</summary>
         *
         * <param name="pruneSet">{@link InstanceList} to perform pruning.</param>
         */
        public void Prune(InstanceList.InstanceList pruneSet)
        {
            Root.Prune(this, pruneSet);
        }

        /**
         * <summary> Training algorithm for C4.5 univariate decision tree classifier. 20 percent of the data are left aside for pruning
         * 80 percent of the data is used for constructing the tree.</summary>
         *
         * <param name="trainSet">  Training data given to the algorithm.</param>
         * <param name="parameters">-</param>
         */
        public override void Train(InstanceList.InstanceList trainSet, Parameter.Parameter parameters)
        {
            if (((C45Parameter)parameters).IsPrune())
            {
                var partition = trainSet.StratifiedPartition(
                    ((C45Parameter)parameters).GetCrossValidationRatio(), new Random(parameters.GetSeed()));
                Root = new DecisionNode(partition.Get(1), null, null, false);
                Prune(partition.Get(0));
            }
            else
            {
                Root = new DecisionNode(trainSet, null, null, false);
            }
        }

        /// <summary>
        /// Loads the decision tree model from an input file.
        /// </summary>
        /// <param name="fileName">File name of the decision tree model.</param>
        public override void LoadModel(string fileName)
        {
            Load(fileName);
        }
    }
}