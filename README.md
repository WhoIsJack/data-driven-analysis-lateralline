## Image-Based Data-Driven Analysis of Cellular Architecture

This repository hosts the code developed for and used in the paper entitled `"An Image-Based Data-Driven Analysis of Cellular Architecture in a Developing Tissue" (Hartmann et al., 2020)`.  All code was written by Jonas Hartmann.

- Preprint on bioRxiv: [link](https://www.biorxiv.org/content/10.1101/2020.02.10.941690v1)
- Peer-reviewed version in eLife: [link](https://elifesciences.org/articles/55913)

Note that the primary purpose of this repository is to ensure **reproducibility** of the specific results presented in the paper. We are working to provide a cleaned and simplified python 3 port that is geared toward **reuse**. If you are interested in using our code in your own work, we recommend that you head over to the [python 3 port repository](https://github.com/WhoIsJack/katachi).


### Code Structure

The repository consists of a python module (`katachi`) that provides functions and of a set of jupyter notebooks that apply these functions to the data.

- Structure of the `katachi` module:
	- `katachi.utilities` provides a number of small modules/functions that perform "atomic" tasks or handle auxiliary tasks like data loading.
	- `katachi.tools` combines the atomic functions in `utilities` into scripts that perform a compound task such as segmentation, ISLA-based point cloud assignment, or CBE feature embedding. Generally, a tool operates on a single sample, though this is not always true.
	- `katachi.pipelines` handles the application of a tool or set of tools to a large dataset. Parallelization is implemented at this level using the [dask](https://dask.org/) library.


- The different types of notebooks:
	- `RUN_*` notebooks run pipelines on input data and save output data. They effectively act as a way of documenting the exact parameters used and the exact sequence of steps employed to get from raw to processed data.
	- `ANALYSIS_*` notebooks load processed data and perform various analyses as well as visualizations of the results.
	- `DEV_*` notebooks were used to optimize and test machine learning steps before commiting to a particular implementation in the corresponding `RUN_*` notebook.


### Data Availability

- Data is available from the [IDR Repository](https://idr.openmicroscopy.org/) under the identifier [idr0079](https://doi.org/10.17867/10000138).
- The IDR provides raw images and segmentations as well as extracted and predicted features.
- The code in this repository expects the IDR data to be present in a folder called `data`.
	- The `ANALYSIS_*` notebooks can be run with only the data provided in the `extracted_measurements` folder in the IDR dataset. If you only want to look at the analysis, it is recommended to only download those folders rather than the entire dataset, since the image data itself is rather large.
	- To recreate the results from raw data (using the `RUN_*` notebooks), the full IDR image data is required.


### Workflow

- Starting from raw data, code needs to be run in the following sequence:
	1. Conversion from 16bit to 8bit images using the ImageJ macro `other\ImageJ_macro_8bit.ijm`
		- The raw data on the IDR has already been converted to 8bit, so this does not need to be run.
    2. `RUN_Initialization.ipynb` creates a file to track metadata and assigns a unique ID to each sample
    3. `RUN_Segmentation.ipynb` performs single-cell segmentation based on labeled cell membranes
    	- The IDR already provides these segmentations, so rerunning it is optinal.
	4. `RUN_FeatureEmbedding.ipynb` performs ISLA and CBE (in both TFOR and CFOR) on segmented cells
	5. `RUN_FeatureEngineering.ipynb` extracts engineered features from images and point clouds (in both TFOR and CFOR)
	6. `RUN_Atlas.ipynb` performs multivariate-multivariable regression to predict embedded features of secondary channels from embedded shape features
	7. `RUN_Archetypes.ipynb` classifies cells into morphological archetypes for context-driven visualization and analysis
	8. For the smFISH dataset (experimentB in the IDR data): `RUN_SpotDetection.ipynb` detects and counts smFISH spots


- The `ANALYSIS_*` notebooks can be run in any order.
	- Since they mostly use the extracted measurements already provided via the IDR, the `RUN_*` notebooks do not need to be run before running any analysis (with a couple of exceptions note directly in the notebooks).
	- If the `RUN_*` notebooks were run and you want to use the results in an `ANALYSIS_*` notebook, you must adjust the data loading in the `ANALYSIS_*` notebook to point at the `image_data` folder instead of the `extracted_measurements` folder, since `RUN_*` notebooks save their results there rather than overwriting the results provided in the IDR. Furthermore, you must use `DataLoader` instead of `DataLoaderIDR`, as the data format on the IDR is slightly different from the original format produced by `RUN_*` notebooks.


### Dependencies

- We recommend installing the [Anaconda distribution](https://www.anaconda.com/products/individual) of python and then using the file `other\conda_env_spec_file.txt` as explained [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#building-identical-conda-environments) to reconstruct the environment used at the time of publication of the code.


- Alternatively, the following key dependencies (and all their dependencies) must be installed:
	- python 2.7.13
	- numpy 1.11.3
	- pandas 0.19.2
	- scipy 1.0.0
	- matplotlib 1.5.1
	- scikit-image 0.13.0
	- scikit-learn 0.19.1
	- seaborn 0.7.1
	- networkx 1.11
	- tifffile 0.11.1
	- dask 0.15.4
	- jupyter 1.0.0
	- notebook 5.3.1


-  Note that a Windows 7 workstation was used in the study and the code has not been tested on any other system.


### Contact

- For any questions or issues pertaining to **reproducibility** of the results presented in the paper, either open an issue in this GitHub repository or contact ![email jh](other/email_JH.png)


- For any questions or issues pertaining **reuse** of the code in a different project or **feature requests**, either open an issue in the [python 3 port repository](https://github.com/WhoIsJack/katachi) or also contact ![email jh](other/email_JH.png)

