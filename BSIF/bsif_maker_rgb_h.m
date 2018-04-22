pkg load image;

in_put = '../data/BBPP/';
out_put = '';

filename='./texturefilters/ICAtextureFilters_17x17_8bit';
load(filename, 'ICAtextureFilters');

vector_feature_length = (2^size(ICAtextureFilters,3))*3;

class_list = dir([in_put '0*']);

for class_index = 1:numel(class_list)


    class_name = class_list(class_index).name;
    image_list = dir([in_put class_name '/*.jpg']);

    for image_index = 1:numel(image_list)

        ID(image_index,1) = {class_list(class_index).name};
        image_name(image_index,1) = {image_list(image_index).name};
        disp(image_list(image_index).name);

        rgb_image = imread([in_put '/' class_name '/' image_list(image_index).name]);

        transformation_type = 'h';

        t=cputime;
        if ndims(rgb_image) == 2
            gray_img = rgb_image;
            r_features = bsif(gray_img, ICAtextureFilters, transformation_type);
            g_features = r_features;
            b_features = r_features;
        elseif ndims(rgb_image) == 3
            r = rgb_image(:,:,1);
            g = rgb_image(:,:,2);
            b = rgb_image(:,:,3);
            r_features = bsif(r, ICAtextureFilters, transformation_type);
            g_features = bsif(g, ICAtextureFilters, transformation_type);
            b_features = bsif(b, ICAtextureFilters, transformation_type);
        endif
        printf('Total cpu time: %f seconds\n', cputime-t);

        % the image with grayvalues replaced by the BSIF bitstrings
        % bsifcodeim= bsif(gray_img,ICAtextureFilters,'im');

        % unnormalized BSIF code word histogram
        % bsifhist=bsif(gray_img,ICAtextureFilters,'h');

        % normalized BSIF code word histogram

        bsifhistnorm(image_index,1:vector_feature_length) = [r_features g_features b_features];

        % class label
        class_index -1
        bsifhistnorm(image_index,vector_feature_length+1)=class_index-1;

        % jako plik matlaba
        %save([out_put '/' image_name '.mat'], bsifhist)
    end
    save("-append", "bsifhistnorm_features_h_rgb.mat", "bsifhistnorm");
end

