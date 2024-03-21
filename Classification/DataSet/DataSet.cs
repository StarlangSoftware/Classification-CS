using System;
using System.Collections.Generic;
using System.IO;
using Classification.Attribute;
using Classification.FeatureSelection;
using Classification.Instance;

namespace Classification.DataSet
{
    public class DataSet
    {
        private readonly InstanceList.InstanceList _instances;
        private DataDefinition _definition;

        /**
         * <summary> Constructor for generating a new {@link DataSet}.</summary>
         */
        public DataSet()
        {
            _definition = null;
            _instances = new InstanceList.InstanceList();
        }

        /**
         * <summary> Constructor for generating a new {@link DataSet} with given {@link DataDefinition}.</summary>
         *
         * <param name="definition">Data definition of the data set.</param>
         */
        public DataSet(DataDefinition definition)
        {
            this._definition = definition;
            _instances = new InstanceList.InstanceList();
        }

        /**
         * <summary> Constructor for generating a new {@link DataSet} from given {@link File}.</summary>
         *
         * <param name="fileName">{@link File} to generate {@link DataSet} from.</param>
         */
        public DataSet(string fileName)
        {
            var i = 0;
            _instances = new InstanceList.InstanceList();
            _definition = new DataDefinition();
            var input = new StreamReader(fileName);
            while (!input.EndOfStream)
            {
                var instanceText = input.ReadLine();
                var attributes = instanceText.Split(",");
                if (i == 0)
                {
                    for (var j = 0; j < attributes.Length - 1; j++)
                    {
                        try
                        {
                            double.Parse(attributes[j]);
                            _definition.AddAttribute(AttributeType.CONTINUOUS);
                        }
                        catch (Exception)
                        {
                            _definition.AddAttribute(AttributeType.DISCRETE);
                        }
                    }
                }
                else
                {
                    if (attributes.Length != _definition.AttributeCount() + 1)
                    {
                        continue;
                    }
                }

                Instance.Instance instance;
                if (!attributes[attributes.Length - 1].Contains(";"))
                {
                    instance = new Instance.Instance(attributes[attributes.Length - 1]);
                }
                else
                {
                    var labels = attributes[attributes.Length - 1].Split(";");
                    instance = new CompositeInstance(labels);
                }

                for (var j = 0; j < attributes.Length - 1; j++)
                {
                    switch (_definition.GetAttributeType(j))
                    {
                        case AttributeType.CONTINUOUS:
                            instance.AddAttribute(new ContinuousAttribute(double.Parse(attributes[j])));
                            break;
                        case AttributeType.DISCRETE:
                            instance.AddAttribute(new DiscreteAttribute(attributes[j]));
                            break;
                    }
                }

                if (instance.AttributeSize() == _definition.AttributeCount())
                {
                    _instances.Add(instance);
                }

                i++;
            }

            input.Close();
        }

        /**
         * <summary> Constructor for generating a new {@link DataSet} with a {@link DataDefinition}, from a {@link File} by using a separator.</summary>
         *
         * <param name="definition">Data definition of the data set.</param>
         * <param name="separator"> Separator character which separates the attribute values in the data file.</param>
         * <param name="fileName">  Name of the data set file.</param>
         */
        public DataSet(DataDefinition definition, string separator, string fileName)
        {
            this._definition = definition;
            _instances = new InstanceList.InstanceList(definition, separator, fileName);
        }

        /**
         * <summary> Checks the correctness of the attribute type, for instance, if the attribute of given instance is a Binary attribute,
         * and the attribute type of the corresponding item of the data definition is also a Binary attribute, it then
         * returns true, and false otherwise.</summary>
         *
         * <param name="instance">{@link Instance} to checks the attribute type.</param>
         * <returns>true if attribute types of given {@link Instance} and data definition matches.</returns>
         */
        private bool CheckDefinition(Instance.Instance instance)
        {
            for (var i = 0; i < instance.AttributeSize(); i++)
            {
                if (instance.GetAttribute(i) is BinaryAttribute)
                {
                    if (_definition.GetAttributeType(i) != AttributeType.BINARY)
                        return false;
                }
                else
                {
                    if (instance.GetAttribute(i) is DiscreteIndexedAttribute)
                    {
                        if (_definition.GetAttributeType(i) != AttributeType.DISCRETE_INDEXED)
                            return false;
                    }
                    else
                    {
                        if (instance.GetAttribute(i) is DiscreteAttribute)
                        {
                            if (_definition.GetAttributeType(i) != AttributeType.DISCRETE)
                                return false;
                        }
                        else
                        {
                            if (instance.GetAttribute(i) is ContinuousAttribute)
                            {
                                if (_definition.GetAttributeType(i) != AttributeType.CONTINUOUS)
                                    return false;
                            }
                        }
                    }
                }
            }

            return true;
        }

        /**
         * <summary> Adds the attribute types according to given {@link Instance}. For instance, if the attribute type of given {@link Instance}
         * is a Discrete type, it than adds a discrete attribute type to the list of attribute types.</summary>
         *
         * <param name="instance">{@link Instance} input.</param>
         */
        private void SetDefinition(Instance.Instance instance)
        {
            var attributeTypes = new List<AttributeType>();
            for (var i = 0; i < instance.AttributeSize(); i++)
            {
                if (instance.GetAttribute(i) is BinaryAttribute)
                {
                    attributeTypes.Add(AttributeType.BINARY);
                }
                else
                {
                    if (instance.GetAttribute(i) is DiscreteIndexedAttribute)
                    {
                        attributeTypes.Add(AttributeType.DISCRETE_INDEXED);
                    }
                    else
                    {
                        if (instance.GetAttribute(i) is DiscreteAttribute)
                        {
                            attributeTypes.Add(AttributeType.DISCRETE);
                        }
                        else
                        {
                            if (instance.GetAttribute(i) is ContinuousAttribute)
                            {
                                attributeTypes.Add(AttributeType.CONTINUOUS);
                            }
                        }
                    }
                }
            }

            _definition = new DataDefinition(attributeTypes);
        }

        /**
         * <summary> Returns the size of the {@link InstanceList}.</summary>
         *
         * <returns>Size of the {@link InstanceList}.</returns>
         */
        public int SampleSize()
        {
            return _instances.Size();
        }

        /**
         * <summary> Returns the size of the class label distribution of {@link InstanceList}.</summary>
         *
         * <returns>Size of the class label distribution of {@link InstanceList}.</returns>
         */
        public int ClassCount()
        {
            return _instances.ClassDistribution().Count;
        }

        /**
         * <summary> Returns the number of attribute types at {@link DataDefinition} list.</summary>
         *
         * <returns>The number of attribute types at {@link DataDefinition} list.</returns>
         */
        public int AttributeCount()
        {
            return _definition.AttributeCount();
        }

        /**
         * <summary> Returns the number of discrete attribute types at {@link DataDefinition} list.</summary>
         *
         * <returns>The number of discrete attribute types at {@link DataDefinition} list.</returns>
         */
        public int DiscreteAttributeCount()
        {
            return _definition.DiscreteAttributeCount();
        }

        /**
         * <summary> Returns the number of continuous attribute types at {@link DataDefinition} list.</summary>
         *
         * <returns>The number of continuous attribute types at {@link DataDefinition} list.</returns>
         */
        public int ContinuousAttributeCount()
        {
            return _definition.ContinuousAttributeCount();
        }

        /**
         * <summary> Returns the accumulated {@link String} of class labels of the {@link InstanceList}.</summary>
         *
         * <returns>The accumulated {@link String} of class labels of the {@link InstanceList}.</returns>
         */
        public string GetClasses()
        {
            var classLabels = _instances.GetDistinctClassLabels();
            var result = classLabels[0];
            for (var i = 1; i < classLabels.Count; i++)
            {
                result = result + ";" + classLabels[i];
            }

            return result;
        }

        /**
         * <summary> Returns the general information about the given data set such as the number of instances, distinct class labels,
         * attributes, discrete and continuous attributes.</summary>
         *
         * <param name="dataSetName">Data set name.</param>
         * <returns>General information about the given data set.</returns>
         */
        public string Info(string dataSetName)
        {
            var result = "DATASET: " + dataSetName + "\n";
            result = result + "Number of instances: " + SampleSize() + "\n";
            result = result + "Number of distinct class labels: " + ClassCount() + "\n";
            result = result + "Number of attributes: " + AttributeCount() + "\n";
            result = result + "Number of discrete attributes: " + DiscreteAttributeCount() + "\n";
            result = result + "Number of continuous attributes: " + ContinuousAttributeCount() + "\n";
            result = result + "Class labels: " + GetClasses();
            return result;
        }

        /**
         * <summary> Returns a formatted String of general information aboutt he data set.</summary>
         *
         * <param name="dataSetName">Data set name.</param>
         * <returns>Formatted String of general information aboutt he data set.</returns>
         */
        public string ToString(string dataSetName)
        {
            return string.Format("{0}{1:F15}{2:F15}{3:F20}{4:F15}{5:F15}", dataSetName, SampleSize(), ClassCount(),
                AttributeCount(),
                DiscreteAttributeCount(), ContinuousAttributeCount());
        }

        /**
         * <summary> Adds a new instance to the {@link InstanceList}.</summary>
         *
         * <param name="current">{@link Instance} to add.</param>
         */
        public void AddInstance(Instance.Instance current)
        {
            if (_definition == null)
            {
                SetDefinition(current);
                _instances.Add(current);
            }
            else
            {
                if (CheckDefinition(current))
                {
                    _instances.Add(current);
                }
            }
        }

        /**
         * <summary> Adds all the instances of given instance list to the {@link InstanceList}.</summary>
         *
         * <param name="instanceList">{@link InstanceList} to add instances from.</param>
         */
        public void AddInstanceList(List<Instance.Instance> instanceList)
        {
            foreach (var instance in instanceList)
            {
                AddInstance(instance);
            }
        }

        /**
         * <summary> Returns the instances of {@link InstanceList}.</summary>
         *
         * <returns>The instances of {@link InstanceList}.</returns>
         */
        public List<Instance.Instance> GetInstances()
        {
            return _instances.GetInstances();
        }

        /**
         * <summary> Returns instances of the items at the list of instance lists from the partitions.</summary>
         *
         * <returns>Instances of the items at the list of instance lists from the partitions.</returns>
         */
        public List<Instance.Instance>[] GetClassInstances()
        {
            return _instances.DivideIntoClasses().GetLists();
        }

        /**
         * <summary> Accessor for the {@link InstanceList}.</summary>
         *
         * <returns>The {@link InstanceList}.</returns>
         */
        public InstanceList.InstanceList GetInstanceList()
        {
            return _instances;
        }

        /**
         * <summary> Accessor for the data definition.</summary>
         *
         * <returns>The data definition.</returns>
         */
        public DataDefinition GetDataDefinition()
        {
            return _definition;
        }

        /**
         * <summary>Return a subset generated via the given {@link FeatureSubSet}.</summary>
         *
         * <param name="featureSubSet">{@link FeatureSubSet} input.</param>
         * <returns>Subset generated via the given {@link FeatureSubSet}.</returns>
         */
        public DataSet GetSubSetOfFeatures(FeatureSubSet featureSubSet)
        {
            var result = new DataSet(_definition.GetSubSetOfFeatures(featureSubSet));
            for (var i = 0; i < _instances.Size(); i++)
            {
                result.AddInstance(_instances.Get(i).GetSubSetOfFeatures(featureSubSet));
            }

            return result;
        }

        /**
         * <summary> Print out the instances of {@link InstanceList} as a {@link String}.</summary>
         *
         * <param name="outFileName">File name to write the output.</param>
         */
        public void WriteToFile(string outFileName)
        {
            var streamWriter = new StreamWriter(File.Create(outFileName));
            for (var i = 0; i < _instances.Size(); i++)
            {
                streamWriter.WriteLine(_instances.Get(i).ToString());
            }

            streamWriter.Close();
        }
    }
}