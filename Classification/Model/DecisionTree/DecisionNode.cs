using System;
using System.Collections.Generic;
using System.IO;
using Classification.Attribute;
using Classification.Instance;
using Classification.Parameter;
using Math;
using Util;

namespace Classification.Model.DecisionTree
{
    public class DecisionNode
    {
        private List<DecisionNode> _children;
        private readonly string _classLabel;
        private bool _leaf;
        private readonly DecisionCondition _condition;
        private DiscreteDistribution _classLabelsDistribution;

        /**
         * <summary> The DecisionNode method takes {@link InstanceList} data as input and then it sets the class label parameter by finding
         * the most occurred class label of given data, it then gets distinct class labels as class labels List. Later, it adds ordered
         * indices to the indexList and shuffles them randomly. Then, it gets the class distribution of given data and finds the best entropy value
         * of these class distribution.
         * <p/>
         * If an attribute of given data is {@link DiscreteIndexedAttribute}, it creates a Distribution according to discrete indexed attribute class distribution
         * and finds the entropy. If it is better than the last best entropy it reassigns the best entropy, best attribute and best split value according to
         * the newly founded best entropy's index. At the end, it also add new distribution to the class distribution .
         * <p/>
         * If an attribute of given data is {@link DiscreteAttribute}, it directly finds the entropy. If it is better than the last best entropy it
         * reassigns the best entropy, best attribute and best split value according to the newly founded best entropy's index.
         * <p/>
         * If an attribute of given data is {@link ContinuousAttribute}, it creates two distributions; left and right according to class distribution
         * and discrete distribution respectively, and finds the entropy. If it is better than the last best entropy it reassigns the best entropy,
         * best attribute and best split value according to the newly founded best entropy's index. At the end, it also add new distribution to
         * the right distribution and removes from left distribution.</summary>
         *
         * <param name="data">     {@link InstanceList} input.</param>
         * <param name="condition">{@link DecisionCondition} to check.</param>
         * <param name="parameter">RandomForestParameter like seed, ensembleSize, attributeSubsetSize.</param>
         * <param name="isStump">  Refers to decision trees with only 1 splitting rule.</param>
         */
        public DecisionNode(InstanceList.InstanceList data, DecisionCondition condition,
            RandomForestParameter parameter,
            bool isStump)
        {
            int bestAttribute = -1, size;
            double bestSplitValue = 0;
            _condition = condition;
            _classLabelsDistribution = new DiscreteDistribution();
            var labels = data.GetClassLabels();
            foreach (var label in labels){
                _classLabelsDistribution.AddItem(label);
            }
            _classLabel = Model.GetMaximum(labels);
            _leaf = true;
            var classLabels = data.GetDistinctClassLabels();
            if (classLabels.Count == 1)
            {
                return;
            }

            if (isStump && condition != null)
            {
                return;
            }

            List<int> indexList;

            if (parameter != null && parameter.GetAttributeSubsetSize() < data.Get(0).AttributeSize())
            {
                indexList = RandomArray.IndexArray(data.Get(0).AttributeSize(), new Random(parameter.GetSeed()));
                size = parameter.GetAttributeSubsetSize();
            }
            else
            {
                indexList = new List<int>();
                for (var i = 0; i < data.Get(0).AttributeSize(); i++)
                {
                    indexList.Add(i);
                }
                size = data.Get(0).AttributeSize();
            }

            var classDistribution = data.ClassDistribution();
            var bestEntropy = data.ClassDistribution().Entropy();
            for (var j = 0; j < size; j++)
            {
                var index = indexList[j];
                double entropy;
                if (data.Get(0).GetAttribute(index) is DiscreteIndexedAttribute)
                {
                    for (var k = 0; k < ((DiscreteIndexedAttribute) data.Get(0).GetAttribute(index)).GetMaxIndex(); k++)
                    {
                        var distribution = data.DiscreteIndexedAttributeClassDistribution(index, k);
                        if (distribution.GetSum() > 0)
                        {
                            classDistribution.RemoveDistribution(distribution);
                            entropy = (classDistribution.Entropy() * classDistribution.GetSum() +
                                       distribution.Entropy() * distribution.GetSum()) / data.Size();
                            if (entropy < bestEntropy)
                            {
                                bestEntropy = entropy;
                                bestAttribute = index;
                                bestSplitValue = k;
                            }

                            classDistribution.AddDistribution(distribution);
                        }
                    }
                }
                else
                {
                    if (data.Get(0).GetAttribute(index) is DiscreteAttribute)
                    {
                        entropy = EntropyForDiscreteAttribute(data, index);
                        if (entropy < bestEntropy)
                        {
                            bestEntropy = entropy;
                            bestAttribute = index;
                        }
                    }
                    else
                    {
                        if (data.Get(0).GetAttribute(index) is ContinuousAttribute)
                        {
                            data.Sort(index);
                            var previousValue = double.MinValue;
                            var leftDistribution = data.ClassDistribution();
                            var rightDistribution = new DiscreteDistribution();
                            for (var k = 0; k < data.Size(); k++)
                            {
                                var instance = data.Get(k);
                                if (k == 0)
                                {
                                    previousValue = ((ContinuousAttribute) instance.GetAttribute(index)).GetValue();
                                }
                                else
                                {
                                    if (((ContinuousAttribute) instance.GetAttribute(index)).GetValue() !=
                                        previousValue)
                                    {
                                        var splitValue =
                                            (previousValue + ((ContinuousAttribute) instance.GetAttribute(index))
                                                .GetValue()) / 2;
                                        previousValue = ((ContinuousAttribute) instance.GetAttribute(index)).GetValue();
                                        entropy =
                                            (leftDistribution.GetSum() / data.Size()) * leftDistribution.Entropy() +
                                            (rightDistribution.GetSum() / data.Size()) * rightDistribution.Entropy();
                                        if (entropy < bestEntropy)
                                        {
                                            bestEntropy = entropy;
                                            bestSplitValue = splitValue;
                                            bestAttribute = index;
                                        }
                                    }
                                }

                                leftDistribution.RemoveItem(instance.GetClassLabel());
                                rightDistribution.AddItem(instance.GetClassLabel());
                            }
                        }
                    }
                }
            }

            if (bestAttribute != -1)
            {
                _leaf = false;
                if (data.Get(0).GetAttribute(bestAttribute) is DiscreteIndexedAttribute)
                {
                    CreateChildrenForDiscreteIndexed(data, bestAttribute, (int) bestSplitValue, parameter, isStump);
                }
                else
                {
                    if (data.Get(0).GetAttribute(bestAttribute) is DiscreteAttribute)
                    {
                        CreateChildrenForDiscrete(data, bestAttribute, parameter, isStump);
                    }
                    else
                    {
                        if (data.Get(0).GetAttribute(bestAttribute) is ContinuousAttribute)
                        {
                            CreateChildrenForContinuous(data, bestAttribute, bestSplitValue, parameter, isStump);
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Reads the decision node model (as one line) from model file.
        /// </summary>
        /// <param name="input">Model file</param>
        public DecisionNode(StreamReader input)
        {
            var line = input.ReadLine();
            var items = line.Split(" ");
            if (!items[0].Equals("-1")){
                if (items[1][0] == '='){
                    _condition = new DecisionCondition(int.Parse(items[0]), items[1][0], new DiscreteAttribute(items[2]));
                } else {
                    if (items[1][0] == ':'){
                        _condition = new DecisionCondition(int.Parse(items[0]), '=', new DiscreteIndexedAttribute("", int.Parse(items[2]), int.Parse(items[3])));
                    } else {
                        _condition = new DecisionCondition(int.Parse(items[0]), items[1][0], new ContinuousAttribute(Double.Parse(items[2])));
                    }
                }
            } else {
                _condition = null;
            }
            var numberOfChildren = int.Parse(input.ReadLine());
            if (numberOfChildren != 0){
                _leaf = false;
                _children = new List<DecisionNode>();
                for (var i = 0; i < numberOfChildren; i++){
                    _children.Add(new DecisionNode(input));
                }
            } else {
                _leaf = true;
                _classLabel = input.ReadLine();
                _classLabelsDistribution = Model.LoadDiscreteDistribution(input);
            }
        }
        
        /**
         * <summary> The entropyForDiscreteAttribute method takes an attributeIndex and creates an List of DiscreteDistribution.
         * Then loops through the distributions and calculates the total entropy.</summary>
         *
         * <param name="data">Instance list</param>
         * <param name="attributeIndex">Index of the attribute.</param>
         * <returns>Total entropy for the discrete attribute.</returns>
         */
        private double EntropyForDiscreteAttribute(InstanceList.InstanceList data, int attributeIndex)
        {
            var sum = 0.0;
            var distributions = data.AttributeClassDistribution(attributeIndex);
            foreach (var distribution in distributions)
            {
                sum += (distribution.GetSum() / data.Size()) * distribution.Entropy();
            }

            return sum;
        }

        /**
         * <summary> The createChildrenForDiscreteIndexed method creates an List of DecisionNodes as children and a partition with respect to
         * indexed attribute.</summary>
         *
         * <param name="data">Instance list</param>
         * <param name="attributeIndex">Index of the attribute.</param>
         * <param name="attributeValue">Value of the attribute.</param>
         * <param name="parameter">     RandomForestParameter like seed, ensembleSize, attributeSubsetSize.</param>
         * <param name="isStump">       Refers to decision trees with only 1 splitting rule.</param>
         */
        private void CreateChildrenForDiscreteIndexed(InstanceList.InstanceList data, int attributeIndex, int attributeValue,
            RandomForestParameter parameter, bool isStump)
        {
            var childrenData = data.DivideWithRespectToIndexedAttribute(attributeIndex, attributeValue);
            _children = new List<DecisionNode>
            {
                new DecisionNode(childrenData.Get(0),
                    new DecisionCondition(attributeIndex,
                        new DiscreteIndexedAttribute("", attributeValue,
                            ((DiscreteIndexedAttribute) data.Get(0).GetAttribute(attributeIndex)).GetMaxIndex())),
                    parameter, isStump),
                new DecisionNode(childrenData.Get(1),
                    new DecisionCondition(attributeIndex,
                        new DiscreteIndexedAttribute("", -1,
                            ((DiscreteIndexedAttribute) data.Get(0).GetAttribute(attributeIndex)).GetMaxIndex())),
                    parameter, isStump)
            };
        }

        /**
         * <summary> The createChildrenForDiscrete method creates an List of values, a partition with respect to attributes and an List
         * of DecisionNodes as children.</summary>
         *
         * <param name="data">Instance list</param>
         * <param name="attributeIndex">Index of the attribute.</param>
         * <param name="parameter">     RandomForestParameter like seed, ensembleSize, attributeSubsetSize.</param>
         * <param name="isStump">       Refers to decision trees with only 1 splitting rule.</param>
         */
        private void CreateChildrenForDiscrete(InstanceList.InstanceList data, int attributeIndex, RandomForestParameter parameter, bool isStump)
        {
            var valueList = data.GetAttributeValueList(attributeIndex);
            var childrenData = data.DivideWithRespectToAttribute(attributeIndex);
            _children = new List<DecisionNode>();
            for (var i = 0; i < valueList.Count; i++)
            {
                _children.Add(new DecisionNode(childrenData.Get(i),
                    new DecisionCondition(attributeIndex, new DiscreteAttribute(valueList[i])), parameter,
                    isStump));
            }
        }

        /**
         * <summary> The createChildrenForContinuous method creates an List of DecisionNodes as children and a partition with respect to
         * continuous attribute and the given split value.</summary>
         *
         * <param name="data">Instance list</param>
         * <param name="attributeIndex">Index of the attribute.</param>
         * <param name="parameter">     RandomForestParameter like seed, ensembleSize, attributeSubsetSize.</param>
         * <param name="isStump">       Refers to decision trees with only 1 splitting rule.</param>
         * <param name="splitValue">    Split value is used for partitioning.</param>
         */
        private void CreateChildrenForContinuous(InstanceList.InstanceList data, int attributeIndex, double splitValue, RandomForestParameter parameter,
            bool isStump)
        {
            var childrenData = data.DivideWithRespectToAttribute(attributeIndex, splitValue);
            _children = new List<DecisionNode>
            {
                new DecisionNode(childrenData.Get(0),
                    new DecisionCondition(attributeIndex, '<', new ContinuousAttribute(splitValue)), parameter,
                    isStump),
                new DecisionNode(childrenData.Get(1),
                    new DecisionCondition(attributeIndex, '>', new ContinuousAttribute(splitValue)), parameter, isStump)
            };
        }

        /**
         * <summary> The prune method takes a {@link DecisionTree} and an {@link InstanceList} as inputs. It checks the classification performance
         * of given InstanceList before pruning, i.e making a node leaf, and after pruning. If the after performance is better than the
         * before performance it prune the given InstanceList from the tree.</summary>
         *
         * <param name="tree">    DecisionTree that will be pruned if conditions hold.</param>
         * <param name="pruneSet">Small subset of tree that will be removed from tree.</param>
         */
        public void Prune(DecisionTree tree, InstanceList.InstanceList pruneSet)
        {
            if (_leaf)
                return;
            var before = tree.TestClassifier(pruneSet);
            _leaf = true;
            var after = tree.TestClassifier(pruneSet);
            if (after.GetAccuracy() < before.GetAccuracy())
            {
                _leaf = false;
                foreach (var node in _children)
                {
                    node.Prune(tree, pruneSet);
                }
            }
        }

        /**
         * <summary> The predict method takes an {@link Instance} as input and performs prediction on the DecisionNodes and returns the prediction
         * for that instance.</summary>
         *
         * <param name="instance">Instance to make prediction.</param>
         * <returns>The prediction for given instance.</returns>
         */
        public string Predict(Instance.Instance instance)
        {
            if (instance is CompositeInstance compositeInstance)
            {
                var possibleClassLabels = compositeInstance.GetPossibleClassLabels();
                var distribution = _classLabelsDistribution;
                var predictedClass = distribution.GetMaxItem(possibleClassLabels);
                if (_leaf)
                {
                    return predictedClass;
                }

                foreach (var node in _children)
                {
                    if (node._condition.Satisfy(compositeInstance))
                    {
                        var childPrediction = node.Predict(compositeInstance);
                        if (childPrediction != null)
                        {
                            return childPrediction;
                        }

                        return predictedClass;
                    }
                }

                return predictedClass;
            }

            if (_leaf)
            {
                return _classLabel;
            }

            foreach (var node in _children)
            {
                if (node._condition.Satisfy(instance))
                {
                    return node.Predict(instance);
                }
            }

            return _classLabel;
        }
        
        /// <summary>
        /// Recursive method that returns the posterior probability distribution of a given instance. If the node is a leaf
        /// node, it returns the class label distribution, otherwise it checks in which direction (child node) this instance
        /// is forwarded.
        /// </summary>
        /// <param name="instance">Instance for which the posterior probability distribution is calculated.</param>
        /// <returns>Posterior probability distribution for this instance.</returns>
        public Dictionary<string, double> PredictProbabilityDistribution(Instance.Instance instance)
        {
            if (_leaf)
            {
                return _classLabelsDistribution.GetProbabilityDistribution();
            }

            foreach (var node in _children)
            {
                if (node._condition.Satisfy(instance))
                {
                    return node.PredictProbabilityDistribution(instance);
                }
            }

            return _classLabelsDistribution.GetProbabilityDistribution();
        }
    }
}