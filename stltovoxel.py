#!/usr/bin/env python3
import argparse
import os.path
import io
import sys
import xml.etree.cElementTree as ET
from zipfile import ZipFile
import zipfile

from PIL import Image
import numpy as np

import slice
import stl_reader
import perimeter
from util import arrayToWhiteGreyscalePixel, padVoxelArray

import tqdm


def doExport(inputFilePath, outputFilePath, resolution, size):
    mesh = list(stl_reader.read_stl_verticies(inputFilePath))
    scale, shift, bounding_box = slice.calculateScaleAndShift(mesh, resolution)
    if size:
        size = np.array(size)
        # Here, bounding box is still (x,y,z)
        for i, d in enumerate(['x', 'y', 'z']):
            if size[i] < bounding_box[i]:
                raise ValueError("Supplied size for Dimension {} ({}) is less than computed bounding box ({})!".format(d, size[i], bounding_box[i]))
        print("Overwriting computed size {} with new size {}".format(bounding_box, size), file=sys.stderr)
        # Need to adjust the scale in order to center the image in the frame
        offset = (size - bounding_box) / 2
        # apply scaling
        offset /= scale
        # Add to shift
        shift += offset
        # set new bounding box
        bounding_box = size

    mesh = list(slice.scaleAndShiftMesh(mesh, scale, shift))
    #Note: vol should be addressed with vol[z][x][y]
    vol = np.empty((bounding_box[2], bounding_box[0], bounding_box[1]), dtype=bool)
    for height in tqdm.tqdm(range(bounding_box[2]), desc='Processing Slice'):
        lines = slice.toIntersectingLines(mesh, height)
        prepixel = np.zeros((bounding_box[0], bounding_box[1]), dtype=bool)
        perimeter.linesToVoxels(lines, prepixel)
        vol[height] = prepixel

    if not size:
        # Adds two extra voxels
        # Only needed if bbox is not given explicitly
        vol, bounding_box = padVoxelArray(vol)

    _, outputFileExtension = os.path.splitext(outputFilePath)
    if outputFileExtension == '.png':
        exportPngs(vol, bounding_box, outputFilePath)
    elif outputFileExtension == '.xyz':
        exportXyz(vol, bounding_box, outputFilePath)
    elif outputFileExtension == '.svx':
        exportSvx(vol, bounding_box, outputFilePath, scale, shift)
    elif outputFileExtension == '.mhd':
        exportMhd(vol, bounding_box, outputFilePath, scale)

def exportPngs(voxels, bounding_box, outputFilePath):
    size = str(len(str(bounding_box[2]))+1)
    outputFilePattern, outputFileExtension = os.path.splitext(outputFilePath)
    for height in range(bounding_box[2]):
        img = Image.new('L', (bounding_box[0], bounding_box[1]), 'black')  # create a new black image
        pixels = img.load()
        arrayToWhiteGreyscalePixel(voxels[height], pixels)
        path = (outputFilePattern + "%0" + size + "d.png")%height
        img.save(path)

def exportXyz(voxels, bounding_box, outputFilePath):
    output = open(outputFilePath, 'w')
    for z in range(bounding_box[2]):
        for x in range(bounding_box[0]):
            for y in range(bounding_box[1]):
                if voxels[z][x][y]:
                    output.write('%s %s %s\n'%(x,y,z))
    output.close()

def exportSvx(voxels, bounding_box, outputFilePath, scale, shift):
    size = str(len(str(bounding_box[2]))+1)
    root = ET.Element("grid", attrib={"gridSizeX": str(bounding_box[0]),
                                      "gridSizeY": str(bounding_box[2]),
                                      "gridSizeZ": str(bounding_box[1]),
                                      "voxelSize": str(1.0/scale[0]/1000), #STL is probably in mm, and svx needs meters
                                      "subvoxelBits": "8",
                                      "originX": str(-shift[0]),
                                      "originY": str(-shift[2]),
                                      "originZ": str(-shift[1]),
                                      })
    channels = ET.SubElement(root, "channels")
    channel = ET.SubElement(channels, "channel", attrib={
        "type":"DENSITY",
        "slices":"density/slice%0" + size + "d.png"
    })
    manifest = ET.tostring(root)
    with ZipFile(outputFilePath, 'w', zipfile.ZIP_DEFLATED) as zipFile:
        for height in range(bounding_box[2]):
            img = Image.new('L', (bounding_box[0], bounding_box[1]), 'black')  # create a new black image
            pixels = img.load()
            arrayToWhiteGreyscalePixel(voxels[height], pixels)
            output = io.BytesIO()
            img.save(output, format="PNG")
            zipFile.writestr(("density/slice%0" + size + "d.png")%height, output.getvalue())
        zipFile.writestr("manifest.xml",manifest)

def exportMhd(voxels, bounding_box, outputFilePath, scale):
    import SimpleITK as sitk
    # We get the voxel data in zxy but need to provide it as zyx for sitk.
    voxels = np.swapaxes(voxels, 1, 2)
    spacing = (1 / np.array(scale)).tolist()
    print("MHD voxel size:", voxels.shape, "spacing:", spacing, file=sys.stderr)
    img = sitk.GetImageFromArray(voxels.astype(np.uint8))
    img.SetSpacing(spacing)
    sitk.WriteImage(img, outputFilePath)


def file_choices(choices,fname):
    filename, ext = os.path.splitext(fname)
    if ext == '' or ext not in choices:
        if len(choices) == 1:
            parser.error('%s doesn\'t end with %s'%(fname,choices))
        else:
            parser.error('%s doesn\'t end with one of %s'%(fname,choices))
    return fname

if __name__ == '__main__':
    """
    Slicing is along the z-axis of the image
    """
    parser = argparse.ArgumentParser(description='Convert STL files to voxels')
    parser.add_argument('--scale', '-s', type=int, default=1,
                        help='Scale the image by given factor '
                             '(makes the voxel size smaller and the number of voxels larger). '
                             'The unit of this parameter is voxel units per physical units')
    parser.add_argument('--size', nargs=3, type=int, help='size of the voxel image (x, y, z), overwrites bounding box calculation.')
    parser.add_argument('input', nargs='?', type=lambda s:file_choices(('.stl'),s))
    parser.add_argument('output', nargs='?', type=lambda s:file_choices(('.png', '.xyz', '.svx', '.mhd'),s))
    args = parser.parse_args()
    doExport(args.input, args.output, args.scale, args.size)
