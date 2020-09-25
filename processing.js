var model;

async function loadModel() {
    model = await tf.loadGraphModel('TFJS/model.json');
}

function predictImage() {
    // console.log('processing ...');
    let image = cv.imread(canvas);
    cv.cvtColor(image, image, cv.COLOR_RGBA2GRAY, 0);
    cv.threshold(image, image, 175, 255, cv.THRESH_BINARY);

    // Find the contour
    let contours = new cv.MatVector();
    let hierarchy = new cv.Mat();
    // You can try more different parameters
    cv.findContours(image, contours, hierarchy, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE);

    // Find the bounding rectangle and crop image
    let cnt = contours.get(0);
    let rect = cv.boundingRect(cnt);
    image = image.roi(rect);

    // Image resize
    var height = image.rows;
    var width = image.cols;
    if (height > width) {
        height = 20;
        const scaleFactor = image.rows / height;
        width = Math.round(image.cols / scaleFactor);
    } else {
        width = 20;
        const scaleFactor = image.cols / width;
        height = Math.round(image.rows / scaleFactor);
    }
    let dsize = new cv.Size(width, height);
    cv.resize(image, image, dsize, 0, 0, cv.INTER_AREA);

    const LEFT = Math.ceil(4 + (20 - width)/2);
    const RIGHT = Math.floor(4 + (20 - width)/2);
    const TOP = Math.ceil(4 + (20 - height)/2);
    const BOTTOM = Math.floor(4 + (20 - height)/2);
    // console.log(`Top: ${TOP} Bottom: ${BOTTOM} Left: ${LEFT} Right: ${RIGHT}`);

    // Add Padding
    const BLACK = new cv.Scalar(0, 0, 0, 0);
    cv.copyMakeBorder(image, image, TOP, BOTTOM, LEFT, RIGHT, cv.BORDER_CONSTANT, BLACK);

    // Center of Mass
    cv.findContours(image, contours, hierarchy, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE);
    cnt = contours.get(0);
    const Moments = cv.moments(cnt, false);

    const cx = Moments.m10 / Moments.m00;
    const cy = Moments.m01 / Moments.m00;
    // console.log(`m00: ${Moments.m00} cx: ${cx} cy: ${cy}`);

    // Shifting the COM of image to the center
    const X_SHIFT = Math.round(image.cols/2.0 - cx);
    const Y_SHIFT = Math.round(image.rows/2.0 - cy);

    dsize = new cv.Size(image.rows, image.cols);
    const M = cv.matFromArray(2, 3, cv.CV_64FC1, [1, 0, X_SHIFT, 0, 1, Y_SHIFT]);
    // You can try more different parameters
    cv.warpAffine(image, image, M, dsize, cv.INTER_LINEAR, cv.BORDER_CONSTANT, BLACK);

    // Normalize the pixel values
    let pixelValues = image.data;
    pixelValues = Float32Array.from(pixelValues);
    pixelValues = pixelValues.map(function(item){
        return item / 255.0;
    });
    // console.log(pixelValues);

    const X = tf.tensor([pixelValues]);
    // console.log(`Shape of Tensor: ${X.shape}`);
    // console.log(`dtype of Tensor: ${X.dtype}`);
    const result = model.predict(X);
    result.print();
    // console.log(tf.memory());

    const output = result.dataSync()[0];

    // Testing only
    // const outputCanvas = document.createElement('CANVAS');
    // cv.imshow(outputCanvas, image);
    // document.body.appendChild(outputCanvas);

    // Cleanup
    image.delete();
    contours.delete();
    cnt.delete();
    hierarchy.delete();
    M.delete();

    X.dispose();
    result.dispose();

    return output;
}