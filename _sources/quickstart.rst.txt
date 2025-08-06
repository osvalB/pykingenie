Quickstart
============

.. code-block:: python

   from pykingenie import *

   bli = OctetExperiment('test')
   # Replace with your own folder containing .frd files !!! The .frd files can be exported from the Octet software
   folder = '/home/os/spc_shiny_servers/kineticsApp/appFiles/KinGenie/www/test_bli_folder/'
   files = os.listdir(folder)
   files = [folder + file for file in files]

   files.sort()

   bli.read_sensor_data(files)

   bli.align_association(bli.sensor_names)
   bli.align_dissociation(bli.sensor_names)

   # We use the sensor #8 as the reference sensor for baseline correction. Remember than python indexing starts at 0.
   bli.subtraction(bli.sensor_names[:7], bli.sensor_names[7])

   pyKinetics = KineticsAnalyzer()
   pyKinetics.add_experiment(bli, 'test')

   labels = pyKinetics.get_experiment_properties('sensor_names')
   ids    = pyKinetics.get_experiment_properties('sensor_names_unique')

   # Flatten the lists
   labels = [item for sublist in labels for item in sublist]
   ids    = [item for sublist in ids for item in sublist]

   df = get_plotting_df(ids, labels)

   figure = plot_traces_all(pyKinetics,df)
   figure.write_image('test_bli_traces.png')

   pyKinetics.merge_ligand_conc_df()
   pyKinetics.init_fittings()

   pyKinetics.generate_fittings(pyKinetics.combined_ligand_conc_df)

   fig = plot_association_dissociation(pyKinetics,split_by_smax_id=True)
   fig.write_image('test_bli.png')

   pyKinetics.submit_steady_state_fitting()
   pyKinetics.submit_kinetics_fitting()

   fig = plot_association_dissociation(pyKinetics)
   fig.write_image('test_bli_fitted.png')