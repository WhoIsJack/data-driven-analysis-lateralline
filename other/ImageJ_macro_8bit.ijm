//
// Created on Fri Mar 03 09:46:47 2017
//
// @author:   Jonas Hartmann @ Gilmour group @ EMBL Heidelberg
//
// @descript: This imageJ macro converts single- or multi-channel stacks or time courses from
//            16bit to 8bit. It can be applied automatically to multiple files in a directory.
// 
// @usage:    Choose inputs and settings in the variable assignments at the top of the script.
//            
//            In minval and maxval, you can either specify the minimum and maximum values to use
//            for rescaling prior to 8bit conversion (just put the number) or you can let these
//            values be determined automatically based on the minimum/maximum value that occurs
//            in each input stack (put a negative number to trigger this). For time courses, the
//            first stack will be used for this automatic determination.
//
//            Fixed values should be used if the resulting intensity values are supposed to be 
//            comparable across multiple stacks. Automated determination should only be used if
//            relative intensity information is sufficient for downstream processing, e.g. for
//            cell segmentation based on a cell membrane marker.
//
//            You can combine both options. For example:
//                minval = newArray(-1, 200)
//            This will automatically determine the minimum value for the first channel for each
//            stack and will use 200 as the minimum value for the second channel for all stacks.


// USER INPUT: Loading & saving
input_dir    = "X:/Path/To/Dir";  // Full path to input directory
input_ending = ".czi";            // Input file extension
stack_order  = "XYCZT";           // Stack order; is usually XYCZT
recurse      = 0;                 // Whether to recurse through subfolders (1) or not (0)

// USER INPUT: Conversion settings
minval    = newArray(-1, 10000);  // Minimum value(s), or negative value(s) to trigger automated determination; single integer or array of integers (one per channel)
maxval    = newArray(-1, 25000);  // Maximum value(s), or negative value(s) to trigger automated determination; single integer or array of integers (one per channel)
reorder   = "12";                 // String of integers (one per channel) giving the output channel order
make_gray = 1;                    // Set the output display option to grayscale (1) or leave it as is (0)


// Conversion function
function make_8bit(filepath, stack_order, minval, maxval, reorder, make_gray) {

	// Open stack using Bio-Formats
	run("Bio-Formats Importer", "open='" + filepath + "'autoscale color_mode=Default view=Hyperstack stack_order="+stack_order);

	// Get stack dimensions
	getDimensions(w, h, channels, slices, frames);

	// For each channel...
	// Note: This does more z projects than really necessary but is easier to read...
	for (ch=1; ch<=channels; ch++) {

		// If required: automatically determine minval by minimum projection 
		// Otherwise: use the given minval
		if (minval[ch-1] < 0) {
			run("Z Project...", "projection=[Min Intensity]");
			if (channels > 1) { Stack.setChannel(ch); }
			getRawStatistics(area, mean, stack_min, maxWrong);
			close();
		} else {
			stack_min = minval[ch-1];
		}
		
		// If required: automatically determine maxval by maximum projection 
		// Otherwise: use the given maxval
		if (maxval[ch-1] < 0) {
			run("Z Project...", "projection=[Max Intensity]");
			if (channels > 1) { Stack.setChannel(ch); }
			getRawStatistics(area, mean, minWrong, stack_max);
			close();
		} else {
			stack_max = maxval[ch-1];
		}

		// Set channel
		if (channels > 1) { Stack.setChannel(ch); }
	
		// Set min and max to the desired range
		setMinAndMax(stack_min, stack_max);
	}

	// Convert to 8-bit
	run("8-bit");

	// Reorder channels as needed
	if (channels > 1) {
		run("Arrange Channels...", "new="+reorder);
	}

	// Make gray if needed
	if (make_gray==1) {
		if (channels>1) {
			Stack.setDisplayMode("grayscale");
		} else {
			run("Grays");
		}
	}

	// Save output
	outpath = substring(filepath, 0, lastIndexOf(filepath, input_ending)) + '_8bit.tif';
	saveAs("Tiff", outpath);
	close();
}


// Get list of files to process
if (recurse==1) {
	filelist = newArray();
	filelist = getFileListRecursive(input_dir, filelist);
} else {
	filelist = getFileList(input_dir);
	for (i=0; i<filelist.length; i++) {
		filelist[i] = input_dir + '/' + filelist[i];
	}
}


// Iterate over files
for (i=0; i<filelist.length; i++) {

	// Check if file has correct ending
	if (endsWith(filelist[i], input_ending)) {
		
		// Make 8bit
		make_8bit(filelist[i], stack_order, minval, maxval, reorder, make_gray);
	}
}


// Helper function: recursive file search
function getFileListRecursive(dir, filelist) {
	list = getFileList(dir);
	for (i=0; i<list.length; i++) {
		if (File.isDirectory(dir+"/"+list[i])) {
			filelist = getFileListRecursive(dir+"/"+list[i], filelist);
		}
		else {
			filelist = append(filelist, dir+"/"+list[i]);
		}
	}
	return filelist;		
}

// Helper function: append value to array
function append(arr, value) {
	arr2 = newArray(arr.length+1);
	for (i=0; i<arr.length; i++) {
    	arr2[i] = arr[i];
	}
 	arr2[arr.length] = value;
 	return arr2;
}

