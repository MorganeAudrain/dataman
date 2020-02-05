# DataMan

Formerly a collection of data organization and documentation tools, now a pipeline to prepare open ephys tetrode
recordings for clustering with KlustaKwik.

The general workflow is to convert to .dat, reference, split into tetrodes, detect spikes, calculate features and
call KlustaKwik. Due to legacy, a large .dat file will be generated first and later split. This is of course 
inefficient, but out of scope for current development.

# Installation
It's recommended to install in a conda environment. Grab the optimized `numpy` and `scipy` from conda, the rest can
be installed via pip using `pip install -e .` in the source directory. It should grab the remaining dependencies.

Additionally, a command line hook should be installed to call the main loop with the short `dm`.

# Usage in short
Help can be accessed via `dm -h`, giving a list of subcommands. Accessing help for these works as expected, e.g.
`dm vis -h`. Verbose output with more information on steps taken can be accessed with the `-v` flag and is generally
recommended to aid debugging when encountering issues.

## Visualization
Given a recording directory, we can quickly take a look at the data with `vis`:

`dm vis ~/data/2014-10-30_15-04-50`

![vis screenshot](dataman/resources/vis_64_channels.png?raw=true)

Vis can read open ephys recording directories (containing `.continuous`), or `.dat` files and can be provided with a probe file to test
proposed channel layouts visually.

Additionally to navigation keys with arrow keys and `shift` or `ctrl` modifiers as well as several command line options,
pressing `f` toggles between wideband and a high-pass filtered view. Double clicking prints current view location in the
console.

NOTE: The data is streamed in a separate thread, but the thread synchronization was never really looked into again. So
opening a file most likely will result in flat lines, just zoom in/out with the RMB or move the signal to refresh the 
view.

## Preparation for clustering
Given that the open ephys recording is most likely not in the proper order, a `.prb` layout file specifies channel ordering and lists
dead channels. Alternatively channel grouping can be specified as CLI arguments and further steps will generate new layout files as needed.
If there's a `.prb` file with the same file name stem as the `.dat` file, it's detected and used automatically. 

### Convert
Convert `1xx_CHxx.continuous` files into flat binary .dat file, reordering channels according to probe file. This requires a `settings.xml`
file to be present to identify which id the recording node uses, among other things. 

**NOTE: **Several versions of the open ephys GUI did not create this file when automatic impedance measurement was enabled.
In that case a `settings.xml` file from a recording with identical signal chain can be reused.

Convert from two recordings sets using a layout file, but limit to the first 10 minutes in total (for shorter test files)
`dm conv ~/data/2014-10-30_15-04-50 ~/data/2014-10-30_15-09-54 -l ~/data/subject_id_16.prb -o ~/proc/testing -D 600`

Alternatively, input files can be specified as a line break delimited text file with `.txt` or `.session` file extensions.
`dm conv -v ~/data/session03.txt -l ~/data/session03.prb`

### Average subtraction re-referencing
Create average of good channels and subtract from all channels, overwriting the unreferenced data. The `-Z` flag zeros out dead channels.
This helps making it obvious during further steps which channels are valid, especially for feature generation and clustering.

`dm ref 2014-10-30_15-04-50.dat -Z`

The `--inplace` flag allows to overwrite the file directly to preserve disk space. If this flag is not used, a new `.dat` file will be
generated alongside the original with the `_meanref` label in the filename.

### Spliting into tetrode specific files
Split a multi-tetrode file into channel groups as per the (reordered) probe file.

`dm split 2014-10-30_15-04-50.dat`

### Dectect and extract spikes
Estimate background noise to calculate a channel-specific threshold. 

### Calculate features

### Cluster with KlustaKwik