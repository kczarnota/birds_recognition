in_put = './SET_A/';
out_put = '';

filename='./texturefilters/ICAtextureFilters_17x17_8bit';
load(filename, 'ICAtextureFilters');

vector_feature_length = 2^size(ICAtextureFilters,3);

class_list = dir([in_put '0*']);

for class_index = 1:numel(class_list)
    
    
    class_name = class_list(class_index).name;
    image_list = dir([in_put class_name '/*.jpg']);
    
    for image_index = 1:numel(image_list)
        
        ID(image_index,1) = {class_list(class_index).name;};
        image_name(image_index,1) = {image_list(image_index).name};
        
        img = imread([in_put '/' image_list(image_index).name]);
        gray_img = rgb2gray(img);
        
        % the image with grayvalues replaced by the BSIF bitstrings
        bsifcodeim= bsif(gray_img,ICAtextureFilters,'im');
        
        % unnormalized BSIF code word histogram
        bsifhist=bsif(gray_img,ICAtextureFilters,'h');
        
        % normalized BSIF code word histogram
        bsifhistnorm(image_index,1:vector_feature_length)=bsif(gray_img, ICAtextureFilters,'nh');
        
        % jako plik matlaba
        %save([out_put '/' image_name '.mat'], bsifhist)
        
        
    end
end
T = table(ID, image_name, bsifhistnorm);
writetable(T,'bsifhistnorm_features.csv')