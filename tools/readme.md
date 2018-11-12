# Tools for data
These codes are mainly written for constructing food detecion dataset with voc format from raw dataset which we collected. 

### step for constructing data
1. data cleaning

   check_annotations.py *
   convert_xml_with_oritation.py *

2. build splited canteen dataset
   create_voc_format_from_raw_data.sh *

3. build merged canteen dataset
   create_crosssval_dataset.py *

4. create categories of different dataset.
   statics.py *
