pkg load image;

in_put = '../data/BBPP/';
out_put = '';

filename='./texturefilters/ICAtextureFilters_17x17_8bit';
load(filename, 'ICAtextureFilters');

vector_feature_length = 2^size(ICAtextureFilters,3);

class_list = dir([in_put '0*']);

for class_index = 1:numel(class_list)


    class_name = class_list(class_index).name;
    image_list = dir([in_put class_name '/*.jpg']);

    for image_index = 1:numel(image_list)

        ID(image_index,1) = {class_list(class_index).name};
        image_name(image_index,1) = {image_list(image_index).name};
        disp(image_list(image_index).name);

        gray_img = imread([in_put '/' class_name '/' image_list(image_index).name]);

        if ndims(gray_img) != 2
            gray_img = rgb2gray(gray_img);
        endif

        % the image with grayvalues replaced by the BSIF bitstrings
        % bsifcodeim= bsif(gray_img,ICAtextureFilters,'im');

        % unnormalized BSIF code word histogram
        % bsifhist=bsif(gray_img,ICAtextureFilters,'h');

        % normalized BSIF code word histogram
        bsifhistnorm(image_index,1:vector_feature_length)=bsif(gray_img, ICAtextureFilters,'nh');
        % class label
        bsifhistnorm(image_index,vector_feature_length+1)=class_index - 1;

        % jako plik matlaba
        %save([out_put '/' image_name '.mat'], bsifhist)
    end
    save("-append", "bsifhistnorm_features.mat", "bsifhistnorm");
end

