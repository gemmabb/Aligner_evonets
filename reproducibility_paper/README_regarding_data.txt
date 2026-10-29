Users can run the notebooks provided in reproducibility_paper/experiment_ directly to reproduce the experiments presented in the paper, using the processed datasets available in data/processed. 

The notebooks used to process the four raw datasets into the inputs required by our model are also provided, allowing users to reproduce the data-processing steps if desired. However, since the raw datasets are large, we do not redistribute them. For example, the Mobility dataset consists of 12 files of approximately 500 MB each. Instead, the readme.txt files in the corresponding subfolders of raw/ provide instructions for downloading the raw data. Users who wish to reproduce the preprocessing can follow these instructions and run the corresponding notebooks to obtain the processed datasets in data/processed.

Following, we want to give credit for the four datasets used and distributed here as processed data:
1. C. elegans connectomes: 
Witvliet, D. et al. Connectomes across development reveal principles of brain maturation. Nature 596, 257449 – 261, 10.1038/s41586-021-03778-8 (2021)
Under license GNU GENERAL PUBLIC LICENSE (Version 3, 29 June 2007).

2. Email network:
Godoy-Lorite, A., Guimerà, R. & Sales-Pardo, M. Long-term evolution of email networks: Statistical regularities, predictability and stability of social behaviors. PLoS ONE 11, e0146113, 10.1371/journal.pone.0146113 (2016).
Under license Creative Commons Attribution 4.0 International

3. Mobility network:
Kang, S., Yuhaoand Gao, Liang, Y., Li, M., Rao, J. & Kruse, J. Multiscale dynamic human mobility flow dataset in the u.s. during the covid-19 epidemic. Sci. Data 7, 390, 10.1038/s41597-020-00734-5 (2020).
Under license MIT License (2018)

4. World trade network:
Gaulier, G. & Zignago, S. Baci: International trade database at the product-level. the 1994-2007 version.470
Working Papers 2010-23, CEPII (2010).
Under license Etalab OPEN LICENCE 2.0
