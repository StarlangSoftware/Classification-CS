using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Classification.Attribute;
using Classification.DataSet;
using Classification.Instance;
using Math;
using Sampling;

namespace Classification.InstanceList
{
    public class InstanceList
    {
        private List<Instance.Instance> _list;

        /**
         * <summary> Empty constructor for an instance list. Initializes the instance list with zero instances.</summary>
         */
        public InstanceList()
        {
            _list = new List<Instance.Instance>();
        }

        /**
         * <summary> Constructor for an instance list with a given data definition, data file and a separator character. Each instance
         * must be stored in a separate line separated with the character separator. The last item must be the class label.
         * The function reads the file line by line and for each line; depending on the data definition, that is, type of
         * the attributes, adds discrete and continuous attributes to a new instance. For example, given the data set file
         * <p/>
         * red;1;0.4;true
         * green;-1;0.8;true
         * blue;3;1.3;false
         * <p/>
         * where the first attribute is a discrete attribute, second and third attributes are continuous attributes, the
         * fourth item is the class label.</summary>
         *
         * <param name="definition">Data definition of the data set.</param>
         * <param name="separator"> Separator character which separates the attribute values in the data file.</param>
         * <param name="fileName">  Name of the data set file.</param>
         */
        public InstanceList(DataDefinition definition, string separator, string fileName)
        {
            _list = new List<Instance.Instance>();
            var streamReader = new StreamReader(fileName);
            var line = streamReader.ReadLine();
            while (line != null)
            {
                var attributeList = line.Split(separator);
                if (attributeList.Length == definition.AttributeCount() + 1)
                {
                    var current = new Instance.Instance(attributeList[attributeList.Length - 1]);
                    for (int i = 0; i < attributeList.Length - 1; i++)
                    {
                        switch (definition.GetAttributeType(i))
                        {
                            case AttributeType.DISCRETE:
                                current.AddAttribute(new DiscreteAttribute(attributeList[i]));
                                break;
                            case AttributeType.BINARY:
                                current.AddAttribute(new BinaryAttribute(bool.Parse(attributeList[i])));
                                break;
                            case AttributeType.CONTINUOUS:
                                current.AddAttribute(new ContinuousAttribute(double.Parse(attributeList[i])));
                                break;
                        }
                    }

                    _list.Add(current);
                }

                line = streamReader.ReadLine();
            }
        }

        /**
         * <summary> Empty constructor for an instance list. Initializes the instance list with the given instance list.</summary>
         *
         * <param name="list">New list for the list variable.</param>
         */
        public InstanceList(List<Instance.Instance> list)
        {
            this._list = list;
        }

        /**
         * <summary> Adds instance to the instance list.</summary>
         *
         * <param name="instance">Instance to be added.</param>
         */
        public void Add(Instance.Instance instance)
        {
            _list.Add(instance);
        }

        /**
         * <summary> Adds a list of instances to the current instance list.</summary>
         *
         * <param name="instanceList">List of instances to be added.</param>
         */
        public void AddAll(List<Instance.Instance> instanceList)
        {
            _list = _list.Union(instanceList).ToList();
        }

        /**
         * <summary> Returns size of the instance list.</summary>
         *
         * <returns>Size of the instance list.</returns>
         */
        public int Size()
        {
            return _list.Count;
        }

        /**
         * <summary> Accessor for a single instance with the given index.</summary>
         *
         * <param name="index">Index of the instance.</param>
         * <returns>Instance with index 'index'.</returns>
         */
        public Instance.Instance Get(int index)
        {
            return _list[index];
        }

        /**
         * <summary> Sorts instance list according to the attribute with index 'attributeIndex'.</summary>
         *
         * <param name="attributeIndex">index of the attribute.</param>
         */
        public void Sort(int attributeIndex)
        {
            _list.Sort(new InstanceComparator(attributeIndex));
        }

        /**
         * <summary> Sorts instance list.</summary>
         */
        public void Sort()
        {
            _list.Sort(new InstanceClassComparator());
        }

        /**
         * <summary> Shuffles the instance list.</summary>
         * <param name="seed">Seed is used for random number generation.</param>
         */
        public void Shuffle(int seed)
        {
        }

        /**
         * <summary> Creates a bootstrap sample from the current instance list.</summary>
         *
         * <param name="seed">To create a different bootstrap sample, we need a new seed for each sample.</param>
         * <returns>Bootstrap sample.</returns>
         */
        public Bootstrap<Instance.Instance> Bootstrap(int seed)
        {
            return new Bootstrap<Instance.Instance>(_list, seed);
        }

        /**
         * <summary> Extracts the class labels of each instance in the instance list and returns them in an array of {@link string}.</summary>
         *
         * <returns>An array list of class labels.</returns>
         */
        public List<string> GetClassLabels()
        {
            var classLabels = new List<string>();
            foreach (var instance in _list)
            {
                classLabels.Add(instance.GetClassLabel());
            }

            return classLabels;
        }

        /**
         * <summary> Extracts the class labels of each instance in the instance list and returns them as a set.</summary>
         *
         * <returns>An {@link List} of distinct class labels.</returns>
         */
        public List<string> GetDistinctClassLabels()
        {
            var classLabels = new List<string>();
            foreach (var instance in _list)
            {
                if (!classLabels.Contains(instance.GetClassLabel()))
                {
                    classLabels.Add(instance.GetClassLabel());
                }
            }

            return classLabels;
        }

        /**
         * <summary> Extracts the possible class labels of each instance in the instance list and returns them as a set.</summary>
         *
         * <returns>An {@link List} of distinct class labels.</returns>
         */
        public List<string> GetUnionOfPossibleClassLabels()
        {
            var possibleClassLabels = new List<string>();
            foreach (var instance in _list)
            {
                if (instance is CompositeInstance compositeInstance)
                {
                    foreach (var possibleClassLabel in compositeInstance.GetPossibleClassLabels())
                    {
                        if (!possibleClassLabels.Contains(possibleClassLabel))
                        {
                            possibleClassLabels.Add(possibleClassLabel);
                        }
                    }
                }
                else
                {
                    if (!possibleClassLabels.Contains(instance.GetClassLabel()))
                    {
                        possibleClassLabels.Add(instance.GetClassLabel());
                    }
                }
            }

            return possibleClassLabels;
        }

        /**
         * <summary> Divides the instances in the instance list into partitions so that all instances of a class are grouped in a
         * single partition.</summary>
         *
         * <returns>Groups of instances according to their class labels.</returns>
         */
        public Partition DivideIntoClasses()
        {
            var classLabels = GetDistinctClassLabels();
            var result = new Partition();
            foreach (var classLabel in classLabels)
                result.Add(new InstanceListOfSameClass(classLabel));
            foreach (var instance in _list)
            {
                result.Get(classLabels.IndexOf(instance.GetClassLabel())).Add(instance);
            }

            return result;
        }

        /**
         * <summary> Extracts distinct discrete values of a given attribute as an array of strings.</summary>
         *
         * <param name="attributeIndex">Index of the discrete attribute.</param>
         * <returns>An array of distinct values of a discrete attribute.</returns>
         */
        public List<string> GetAttributeValueList(int attributeIndex)
        {
            var valueList = new List<string>();
            foreach (var instance in _list)
            {
                if (!valueList.Contains(((DiscreteAttribute) instance.GetAttribute(attributeIndex)).GetValue()))
                {
                    valueList.Add(((DiscreteAttribute) instance.GetAttribute(attributeIndex)).GetValue());
                }
            }

            return valueList;
        }

        /**
         * <summary> Creates a stratified partition of the current instance list. In a stratified partition, the percentage of each
         * class is preserved. For example, let's say there are three classes in the instance list, and let the percentages of
         * these classes be %20, %30, and %50; then the percentages of these classes in the stratified partitions are the
         * same, that is, %20, %30, and %50.</summary>
         *
         * <param name="ratio">Ratio of the stratified partition. Ratio is between 0 and 1. If the ratio is 0.2, then 20 percent</param>
         *              of the instances are put in the first group, 80 percent of the instances are put in the second group.
         * <param name="random">random is used as a random number.</param>
         * <returns>2 group stratified partition of the instances in this instance list.</returns>
         */
        public Partition StratifiedPartition(double ratio, Random random)
        {
            Partition partition = new Partition();
            partition.Add(new InstanceList());
            partition.Add(new InstanceList());
            var distribution = ClassDistribution();
            var counts = new int[distribution.Count];
            var randomArray = new List<int>();
            for (var i = 0; i < Size(); i++)
                randomArray.Add(i);
            for (var i = 0; i < Size(); i++)
            {
                var instance = _list[randomArray[i]];
                var classIndex = distribution.GetIndex(instance.GetClassLabel());
                if (counts[classIndex] < Size() * ratio * distribution.GetProbability(instance.GetClassLabel()))
                {
                    partition.Get(0).Add(instance);
                }
                else
                {
                    partition.Get(1).Add(instance);
                }

                counts[classIndex]++;
            }

            return partition;
        }

        /**
         * <summary> Creates a partition of the current instance list.</summary>
         *
         * <param name="ratio">Ratio of the partition. Ratio is between 0 and 1. If the ratio is 0.2, then 20 percent</param>
         *              of the instances are put in the first group, 80 percent of the instances are put in the second group.
         * <param name="random">random is used as a random number.</param>
         * <returns>2 group partition of the instances in this instance list.</returns>
         */
        public Partition Partition(double ratio, Random random)
        {
            var partition = new Partition();
            partition.Add(new InstanceList());
            partition.Add(new InstanceList());
            for (var i = 0; i < Size(); i++)
            {
                var instance = _list[i];
                if (i < Size() * ratio)
                {
                    partition.Get(0).Add(instance);
                }
                else
                {
                    partition.Get(1).Add(instance);
                }
            }

            return partition;
        }

        /**
         * <summary> Creates a partition depending on the distinct values of a discrete attribute. If the discrete attribute has 4
         * distinct values, the resulting partition will have 4 groups, where each group contain instance whose
         * values of that discrete attribute are the same.</summary>
         *
         * <param name="attributeIndex">Index of the discrete attribute.</param>
         * <returns>L groups of instances, where L is the number of distinct values of the discrete attribute with index</returns>
         * attributeIndex.
         */
        public Partition DivideWithRespectToAttribute(int attributeIndex)
        {
            var valueList = GetAttributeValueList(attributeIndex);
            var result = new Partition();
            foreach (var value in valueList)
            {
                result.Add(new InstanceList());
            }

            foreach (var instance in _list)
            {
                result.Get(valueList.IndexOf(((DiscreteAttribute) instance.GetAttribute(attributeIndex)).GetValue()))
                    .Add(instance);
            }

            return result;
        }

        /**
         * <summary> Creates a partition depending on the distinct values of a discrete indexed attribute.</summary>
         *
         * <param name="attributeIndex">Index of the discrete indexed attribute.</param>
         * <param name="attributeValue">Value of the attribute.</param>
         * <returns>L groups of instances, where L is the number of distinct values of the discrete indexed attribute with index
         * attributeIndex and value attributeValue.</returns>
         */
        public Partition DivideWithRespectToIndexedAttribute(int attributeIndex, int attributeValue)
        {
            var result = new Partition();
            result.Add(new InstanceList());
            result.Add(new InstanceList());
            foreach (var instance in _list)
            {
                if (((DiscreteIndexedAttribute) instance.GetAttribute(attributeIndex)).GetIndex() == attributeValue)
                {
                    result.Get(0).Add(instance);
                }
                else
                {
                    result.Get(1).Add(instance);
                }
            }

            return result;
        }

        /**
         * <summary> Creates a two group partition depending on the values of a continuous attribute. If the value of the attribute is
         * less than splitValue, the instance is forwarded to the first group, else it is forwarded to the second group.</summary>
         *
         * <param name="attributeIndex">Index of the continuous attribute</param>
         * <param name="splitValue">    Threshold to divide instances</param>
         * <returns>Two groups of instances as a partition.</returns>
         */
        public Partition DivideWithRespectToAttribute(int attributeIndex, double splitValue)
        {
            var result = new Partition();
            result.Add(new InstanceList());
            result.Add(new InstanceList());
            foreach (var instance in _list)
            {
                if (((ContinuousAttribute) instance.GetAttribute(attributeIndex)).GetValue() <= splitValue)
                {
                    result.Get(0).Add(instance);
                }
                else
                {
                    result.Get(1).Add(instance);
                }
            }

            return result;
        }

        /**
         * <summary> Calculates the mean of a single attribute for this instance list (m_i). If the attribute is discrete, the maximum
         * occurring value for that attribute is returned. If the attribute is continuous, the mean value of the values of
         * all instances are returned.</summary>
         *
         * <param name="index">Index of the attribute.</param>
         * <returns>The mean value of the instances as an attribute.</returns>
         */
        private Attribute.Attribute AttributeAverage(int index)
        {
            if (_list[0].GetAttribute(index) is DiscreteAttribute)
            {
                var values = new List<string>();
                foreach (var instance in _list)
                {
                    values.Add(((DiscreteAttribute) instance.GetAttribute(index)).GetValue());
                }

                return new DiscreteAttribute(Classifier.Classifier.GetMaximum(values));
            }

            if (_list[0].GetAttribute(index) is ContinuousAttribute)
            {
                var sum = 0.0;
                foreach (var instance in _list)
                {
                    sum += ((ContinuousAttribute) instance.GetAttribute(index)).GetValue();
                }

                return new ContinuousAttribute(sum / _list.Count);
            }

            return null;
        }

        /**
         * <summary> Calculates the mean of a single attribute for this instance list (m_i).</summary>
         *
         * <param name="index">Index of the attribute.</param>
         * <returns>The mean value of the instances as an attribute.</returns>
         */
        private List<double> ContinuousAttributeAverage(int index)
        {
            if (_list[0].GetAttribute(index) is DiscreteIndexedAttribute)
            {
                var maxIndexSize = ((DiscreteIndexedAttribute) _list[0].GetAttribute(index)).GetMaxIndex();
                var values = new List<double>();
                for (var i = 0; i < maxIndexSize; i++)
                {
                    values.Add(0.0);
                }

                foreach (var instance in _list)
                {
                    var valueIndex = ((DiscreteIndexedAttribute) instance.GetAttribute(index)).GetIndex();
                    values[valueIndex] = values[valueIndex] + 1;
                }

                for (var i = 0; i < values.Count; i++)
                {
                    values[i] = values[i] / _list.Count;
                }

                return values;
            }

            if (_list[0].GetAttribute(index) is ContinuousAttribute)
            {
                var sum = 0.0;
                foreach (var instance in _list)
                {
                    sum += ((ContinuousAttribute) instance.GetAttribute(index)).GetValue();
                }

                var values = new List<double> {sum / _list.Count};
                return values;
            }

            return null;
        }

        /**
         * <summary> Calculates the standard deviation of a single attribute for this instance list (m_i). If the attribute is discrete,
         * null returned. If the attribute is continuous, the standard deviation  of the values all instances are returned.</summary>
         *
         * <param name="index">Index of the attribute.</param>
         * <returns>The standard deviation of the instances as an attribute.</returns>
         */
        private Attribute.Attribute AttributeStandardDeviation(int index)
        {
            if (_list[0].GetAttribute(index) is ContinuousAttribute)
            {
                var sum = 0.0;
                foreach (var instance in _list)
                {
                    sum += ((ContinuousAttribute) instance.GetAttribute(index)).GetValue();
                }

                var average = sum / _list.Count;
                sum = 0.0;
                foreach (var instance in _list)
                {
                    sum += System.Math.Pow(((ContinuousAttribute) instance.GetAttribute(index)).GetValue() - average,
                        2);
                }

                return new ContinuousAttribute(System.Math.Sqrt(sum / (_list.Count - 1)));
            }

            return null;
        }

        /**
         * <summary> Calculates the standard deviation of a single continuous attribute for this instance list (m_i).</summary>
         *
         * <param name="index">Index of the attribute.</param>
         * <returns>The standard deviation of the instances as an attribute.</returns>
         */
        private List<double> ContinuousAttributeStandardDeviation(int index)
        {
            if (_list[0].GetAttribute(index) is DiscreteIndexedAttribute)
            {
                var maxIndexSize = ((DiscreteIndexedAttribute) _list[0].GetAttribute(index)).GetMaxIndex();
                var averages = new List<double>();
                for (var i = 0; i < maxIndexSize; i++)
                {
                    averages.Add(0.0);
                }

                foreach (var instance in _list)
                {
                    var valueIndex = ((DiscreteIndexedAttribute) instance.GetAttribute(index)).GetIndex();
                    averages[valueIndex] = averages[valueIndex] + 1;
                }

                for (var i = 0; i < averages.Count; i++)
                {
                    averages[i] = averages[i] / _list.Count;
                }

                var values = new List<double>();
                for (var i = 0; i < maxIndexSize; i++)
                {
                    values.Add(0.0);
                }

                foreach (var instance in _list)
                {
                    var valueIndex = ((DiscreteIndexedAttribute) instance.GetAttribute(index)).GetIndex();
                    for (var i = 0; i < maxIndexSize; i++)
                    {
                        if (i == valueIndex)
                        {
                            values[i] = values[i] + System.Math.Pow(1 - averages[i], 2);
                        }
                        else
                        {
                            values[i] = values[i] + System.Math.Pow(averages[i], 2);
                        }
                    }
                }

                for (var i = 0; i < values.Count; i++)
                {
                    values[i] = System.Math.Sqrt(values[i] / (_list.Count - 1));
                }

                return values;
            }

            if (_list[0].GetAttribute(index) is ContinuousAttribute)
            {
                var sum = 0.0;
                foreach (var instance in _list)
                {
                    sum += ((ContinuousAttribute) instance.GetAttribute(index)).GetValue();
                }

                var average = sum / _list.Count;
                sum = 0.0;
                foreach (var instance in _list)
                {
                    sum += System.Math.Pow(((ContinuousAttribute) instance.GetAttribute(index)).GetValue() - average,
                        2);
                }

                var result = new List<double> {System.Math.Sqrt(sum / (_list.Count - 1))};
                return result;
            }

            return null;
        }

        /**
         * <summary> The attributeDistribution method takes an index as an input and if the attribute of the instance at given index is
         * discrete, it returns the distribution of the attributes of that instance.</summary>
         *
         * <param name="index">Index of the attribute.</param>
         * <returns>Distribution of the attribute.</returns>
         */
        public DiscreteDistribution AttributeDistribution(int index)
        {
            var distribution = new DiscreteDistribution();
            if (_list[0].GetAttribute(index) is DiscreteAttribute)
            {
                foreach (var instance in _list)
                {
                    distribution.AddItem(((DiscreteAttribute) instance.GetAttribute(index)).GetValue());
                }
            }

            return distribution;
        }

        /**
         * <summary> The attributeClassDistribution method takes an attribute index as an input. It loops through the instances, gets
         * the corresponding value of given attribute index and adds the class label of that instance to the discrete distributions list.</summary>
         *
         * <param name="attributeIndex">Index of the attribute.</param>
         * <returns>Distribution of the class labels.</returns>
         */
        public List<DiscreteDistribution> AttributeClassDistribution(int attributeIndex)
        {
            var distributions = new List<DiscreteDistribution>();
            var valueList = GetAttributeValueList(attributeIndex);
            foreach (var ignored in valueList)
            {
                distributions.Add(new DiscreteDistribution());
            }

            foreach (var instance in _list)
            {
                distributions[valueList.IndexOf(((DiscreteAttribute) instance.GetAttribute(attributeIndex)).GetValue())]
                    .AddItem(instance.GetClassLabel());
            }

            return distributions;
        }

        /**
         * <summary> The discreteIndexedAttributeClassDistribution method takes an attribute index and an attribute value as inputs.
         * It loops through the instances, gets the corresponding value of given attribute index and given attribute value.
         * Then, adds the class label of that instance to the discrete indexed distributions list.</summary>
         *
         * <param name="attributeIndex">Index of the attribute.</param>
         * <param name="attributeValue">Value of the attribute.</param>
         * <returns>Distribution of the class labels.</returns>
         */
        public DiscreteDistribution DiscreteIndexedAttributeClassDistribution(int attributeIndex, int attributeValue)
        {
            var distribution = new DiscreteDistribution();
            foreach (var instance in _list) {
                if (((DiscreteIndexedAttribute) instance.GetAttribute(attributeIndex)).GetIndex() == attributeValue)
                {
                    distribution.AddItem(instance.GetClassLabel());
                }
            }
            return distribution;
        }

        /**
         * <summary> The classDistribution method returns the distribution of all the class labels of instances.</summary>
         *
         * <returns>Distribution of the class labels.</returns>
         */
        public DiscreteDistribution ClassDistribution()
        {
            var distribution = new DiscreteDistribution();
            foreach (var instance in _list) {
                distribution.AddItem(instance.GetClassLabel());
            }
            return distribution;
        }

        /**
         * <summary> The allAttributesDistribution method returns the distributions of all the attributes of instances.</summary>
         *
         * <returns>Distributions of all the attributes of instances.</returns>
         */
        public List<DiscreteDistribution> AllAttributesDistribution()
        {
            var distributions = new List<DiscreteDistribution>();
            for (var i = 0; i < _list[0].AttributeSize(); i++)
            {
                distributions.Add(AttributeDistribution(i));
            }

            return distributions;
        }

        /**
         * <summary> Returns the mean of all the attributes for instances in the list.</summary>
         *
         * <returns>Mean of all the attributes for instances in the list.</returns>
         */
        public Instance.Instance Average()
        {
            Instance.Instance result = new Instance.Instance(_list[0].GetClassLabel());
            for (int i = 0; i < _list[0].AttributeSize(); i++)
            {
                result.AddAttribute(AttributeAverage(i));
            }

            return result;
        }

        /**
         * <summary> Calculates mean of the attributes of instances.</summary>
         *
         * <returns>Mean of the attributes of instances.</returns>
         */
        public List<double> ContinuousAttributeAverage()
        {
            var result = new List<double>();
            for (var i = 0; i < _list[0].AttributeSize(); i++)
            {
                result = result.Union(ContinuousAttributeAverage(i)).ToList();
            }

            return result;
        }

        /**
         * <summary> Returns the standard deviation of attributes for instances.</summary>
         *
         * <returns>Standard deviation of attributes for instances.</returns>
         */
        public Instance.Instance StandardDeviation()
        {
            var result = new Instance.Instance(_list[0].GetClassLabel());
            for (var i = 0; i < _list[0].AttributeSize(); i++)
            {
                result.AddAttribute(AttributeStandardDeviation(i));
            }

            return result;
        }

        /**
         * <summary> Returns the standard deviation of continuous attributes for instances.</summary>
         *
         * <returns>Standard deviation of continuous attributes for instances.</returns>
         */
        public List<double> ContinuousAttributeStandardDeviation()
        {
            var result = new List<double>();
            for (var i = 0; i < _list[0].AttributeSize(); i++)
            {
                result = result.Union(ContinuousAttributeStandardDeviation(i)).ToList();
            }

            return result;
        }

        /**
         * <summary> Calculates a covariance {@link Matrix} by using an average {@link Vector}.</summary>
         *
         * <param name="average">Vector input.</param>
         * <returns>Covariance {@link Matrix}.</returns>
         */
        public Matrix Covariance(Vector average)
        {
            var result = new Matrix(_list[0].ContinuousAttributeSize(), _list[0].ContinuousAttributeSize());
            foreach (var instance in _list) {
                var continuousAttributes = instance.ContinuousAttributes();
                for (var i = 0; i < instance.ContinuousAttributeSize(); i++)
                {
                    var xi = continuousAttributes[i];
                    var mi = average.GetValue(i);
                    for (var j = 0; j < instance.ContinuousAttributeSize(); j++)
                    {
                        var xj = continuousAttributes[j];
                        var mj = average.GetValue(j);
                        result.AddValue(i, j, (xi - mi) * (xj - mj));
                    }
                }
            }
            result.DivideByConstant(_list.Count - 1);
            return result;
        }

        /**
         * <summary> Accessor for the instances.</summary>
         *
         * <returns>Instances.</returns>
         */
        public List<Instance.Instance> GetInstances()
        {
            return _list;
        }
    }
}