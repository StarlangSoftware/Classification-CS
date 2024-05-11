using System.Collections.Generic;
using Classification.Attribute;
using Classification.FeatureSelection;

namespace Classification.DataSet
{
    public class DataDefinition
    {
        private readonly List<AttributeType> _attributeTypes;
        private readonly string[][] _attributeValueList;

        /**
         * <summary> Constructor for creating a new {@link DataDefinition}.</summary>
         */
        public DataDefinition()
        {
            _attributeTypes = new List<AttributeType>();
        }

        /**
         * <summary> Constructor for creating a new {@link DataDefinition} with given attribute types.</summary>
         *
         * <param name="attributeTypes">Attribute types of the data definition.</param>
         */
        public DataDefinition(List<AttributeType> attributeTypes)
        {
            _attributeTypes = attributeTypes;
        }

        /**
         * <summary>Constructor for creating a new <see cref="DataDefinition"/> with given attribute types.</summary>
         *
         * <param name="attributeTypes"> Attribute types of the data definition.</param>
         * <param name="attributeValueList"> Array of array of strings to represent all possible values of discrete features.</param>
         */
        public DataDefinition(List<AttributeType> attributeTypes, string[][] attributeValueList)
        {
            _attributeTypes = attributeTypes;
            _attributeValueList = attributeValueList;
        }

        /// <summary>
        /// Returns number of distinct values for a given discrete attribute with index attributeIndex.
        /// </summary>
        /// <param name="attributeIndex">Index of the discrete attribute.</param>
        /// <returns>Number of distinct values for a given discrete attribute</returns>
        public int NumberOfValues(int attributeIndex){
            return _attributeValueList[attributeIndex].Length;
        }

        /// <summary>
        /// Returns the index of the given value in the values list of the attributeIndex'th discrete attribute.
        /// </summary>
        /// <param name="attributeIndex">Index of the discrete attribute.</param>
        /// <param name="value">Value of the discrete attribute</param>
        /// <returns>Index of the given value in the values list of the discrete attribute.</returns>
        public int FeatureValueIndex(int attributeIndex, string value){
            for (var i = 0; i < _attributeValueList[attributeIndex].Length; i++){
                if (_attributeValueList[attributeIndex][i].Equals(value)){
                    return i;
                }
            }
            return -1;
        }

        /**
         * <summary> Returns the number of attribute types.</summary>
         *
         * <returns>Number of attribute types.</returns>
         */
        public int AttributeCount()
        {
            return _attributeTypes.Count;
        }

        /**
         * <summary> Counts the occurrences of binary and discrete type attributes.</summary>
         *
         * <returns>Count of binary and discrete type attributes.</returns>
         */
        public int DiscreteAttributeCount()
        {
            var count = 0;
            foreach (var attributeType in _attributeTypes)
            {
                if (attributeType.Equals(AttributeType.DISCRETE) || attributeType.Equals(AttributeType.BINARY))
                {
                    count++;
                }
            }

            return count;
        }

        /**
         * <summary> Counts the occurrences of binary and continuous type attributes.</summary>
         *
         * <returns>Count of of binary and continuous type attributes.</returns>
         */
        public int ContinuousAttributeCount()
        {
            var count = 0;
            foreach (var attributeType in _attributeTypes)
            {
                if (attributeType.Equals(AttributeType.CONTINUOUS))
                {
                    count++;
                }
            }

            return count;
        }

        /**
         * <summary> Returns the attribute type of the corresponding item at given index.</summary>
         *
         * <param name="index">Index of the attribute type.</param>
         * <returns>Attribute type of the corresponding item at given index.</returns>
         */
        public AttributeType GetAttributeType(int index)
        {
            return _attributeTypes[index];
        }

        /**
         * <summary> Adds an attribute type to the list of attribute types.</summary>
         *
         * <param name="attributeType">Attribute type to add to the list of attribute types.</param>
         */
        public void AddAttribute(AttributeType attributeType)
        {
            _attributeTypes.Add(attributeType);
        }

        /**
         * <summary> Removes the attribute type at given index from the list of attributes.</summary>
         *
         * <param name="index">Index to remove attribute type from list.</param>
         */
        public void RemoveAttribute(int index)
        {
            _attributeTypes.RemoveAt(index);
        }

        /**
         * <summary> Clears all the attribute types from list.</summary>
         */
        public void RemoveAllAttributes()
        {
            _attributeTypes.Clear();
        }

        /**
         * <summary>Generates new subset of attribute types by using given feature subset.</summary>
         *
         * <param name="featureSubSet">{@link FeatureSubSet} input.</param>
         * <returns>DataDefinition with new subset of attribute types.</returns>
         */
        public DataDefinition GetSubSetOfFeatures(FeatureSubSet featureSubSet)
        {
            var newAttributeTypes = new List<AttributeType>();
            for (var i = 0; i < featureSubSet.Size(); i++)
            {
                newAttributeTypes.Add(_attributeTypes[featureSubSet.Get(i)]);
            }

            return new DataDefinition(newAttributeTypes);
        }
    }
}