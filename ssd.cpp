/* ---------------------------------------------------------------------------
** This software is in the public domain, furnished "as is", without technical
** support, and with no warranty, express or implied, as to its usefulness for
** any purpose.
**
** Author: Joo Saw
**
** -------------------------------------------------------------------------*/

#include <sys/time.h>
#include <iostream>
extern "C" {
#include <libavutil/imgutils.h>
#include <libavutil/parseutils.h>
#include <libswscale/swscale.h>
}
#include "log.hpp"
#include "ssd.hpp"

#include <opencv2/opencv.hpp> // for IMWRITE_PNG_COMPRESSION



namespace ssd {

//const size_t SSDetector::keypointsNumber = 18;


void SSDetector::get_scaled_image_dimensions(int width, int height,
                                             int *scaled_width, int *scaled_height)
{
    int input_height = 0;
    int input_width  = 0;
    getInputWidthHeight(&input_width, &input_height);
    double scale_h = (double)input_height / height;
    double scale_w = (double)input_width  / width;
    double scale   = MIN(scale_h, scale_w);
    *scaled_width  = (int)(width * scale);
    *scaled_height = (int)(height * scale);
}


// caller must av_freep returned image
unsigned char *SSDetector::scale_yuv2bgr(unsigned char *src_img, int width, int height, int scaled_width, int scaled_height)
{
    uint8_t *src_data[4] = {0};
    uint8_t *dst_data[4] = {0};
    int src_linesize[4] = {0};
    int dst_linesize[4] = {0};
    int src_w = width;
    int src_h = height;
    int dst_w = scaled_width;
    int dst_h = scaled_height;
    enum AVPixelFormat src_pix_fmt = AV_PIX_FMT_YUV420P;
    enum AVPixelFormat dst_pix_fmt = AV_PIX_FMT_BGR24;
    struct SwsContext *sws_ctx = NULL;

    // create scaling context
    sws_ctx = sws_getContext(src_w, src_h, src_pix_fmt,
                             dst_w, dst_h, dst_pix_fmt,
                             SWS_BICUBIC, NULL, NULL, NULL);
    if (!sws_ctx)
    {
        std::ostringstream stringStream;
        stringStream << "Impossible to create scale context for image conversion fmt:"
                     << av_get_pix_fmt_name(src_pix_fmt) << " s:" << src_w << "x" << src_h
                     << " -> fmt:" << av_get_pix_fmt_name(dst_pix_fmt) << " s:" << dst_w << "x" << dst_h;
        errMessage = stringStream.str();
        return NULL;
    }

    int srcNumBytes = av_image_fill_arrays(src_data, src_linesize, src_img,
                                           src_pix_fmt, src_w, src_h, 1);
    if (srcNumBytes < 0)
    {
        std::ostringstream stringStream;
        stringStream << "Failed to fill image arrays: code " << srcNumBytes;
        errMessage = stringStream.str();
        sws_freeContext(sws_ctx);
        return NULL;
    }

    int dst_bufsize;
    if ((dst_bufsize = av_image_alloc(dst_data, dst_linesize,
                       dst_w, dst_h, dst_pix_fmt, 1)) < 0)
    {
        std::ostringstream stringStream;
        stringStream << "Failed to allocate dst image";
        errMessage = stringStream.str();
        sws_freeContext(sws_ctx);
        return NULL;
    }

    // convert to destination format
    sws_scale(sws_ctx, (const uint8_t * const*)src_data,
              src_linesize, 0, src_h, dst_data, dst_linesize);

    sws_freeContext(sws_ctx);
    return dst_data[0];
}


inline std::size_t getTensorWidth(const InferenceEngine::TensorDesc& desc) {
    const auto& layout = desc.getLayout();
    const auto& dims = desc.getDims();
    const auto& size = dims.size();
    if ((size >= 2) &&
        (layout == InferenceEngine::Layout::NCHW  ||
         layout == InferenceEngine::Layout::NHWC  ||
         layout == InferenceEngine::Layout::NCDHW ||
         layout == InferenceEngine::Layout::NDHWC ||
         layout == InferenceEngine::Layout::OIHW  ||
         layout == InferenceEngine::Layout::CHW   ||
         layout == InferenceEngine::Layout::HW)) {
        // Regardless of layout, dimensions are stored in fixed order
        return dims.back();
    } else {
        THROW_IE_EXCEPTION << "Tensor does not have width dimension";
    }
    return 0;
}


inline std::size_t getTensorHeight(const InferenceEngine::TensorDesc& desc) {
    const auto& layout = desc.getLayout();
    const auto& dims = desc.getDims();
    const auto& size = dims.size();
    if ((size >= 2) &&
        (layout == InferenceEngine::Layout::NCHW  ||
         layout == InferenceEngine::Layout::NHWC  ||
         layout == InferenceEngine::Layout::NCDHW ||
         layout == InferenceEngine::Layout::NDHWC ||
         layout == InferenceEngine::Layout::OIHW  ||
         layout == InferenceEngine::Layout::CHW   ||
         layout == InferenceEngine::Layout::HW)) {
        // Regardless of layout, dimensions are stored in fixed order
        return dims.at(size - 2);
    } else {
        THROW_IE_EXCEPTION << "Tensor does not have height dimension";
    }
    return 0;
}


SSDetector::SSDetector(bool matchJobIdToWorkerId_,
                       int queueSize_,
                       int numDevices_,
                       const std::string& modelXmlPath,
                       const std::string& modelBinPath,
                       const std::string& targetDeviceName,
                       std::set<int> idFilter_)
    : meanPixel(cv::Vec3f::all(255)),
      inputLayerSize(-1, -1) {

    (void)targetDeviceName;

    matchJobIdToWorkerId = matchJobIdToWorkerId_;
    numDevices = numDevices_;
    idFilter = idFilter_;

    // TODO: set log callback
    //SetLogCallback();

    // get ALL inference devices
    std::vector<std::string> availableDevices = ie.GetAvailableDevices();
    if ((numDevices <= 0) || ((int)availableDevices.size() < numDevices))
        numDevices = availableDevices.size();

    // Debug only
    //std::cout << "Available devices: " << std::endl;
    //for (auto && device : availableDevices) {
    //    std::cout << "\tDevice: " << device << std::endl;
    //}

    // read model
    InferenceEngine::CNNNetReader netReader;
    netReader.ReadNetwork(modelXmlPath); // model.xml file
    netReader.ReadWeights(modelBinPath); // model.bin file

    network = netReader.getNetwork();
    network.setBatchSize(1);

    // prepare input blobs
    InferenceEngine::InputsDataMap input_data_map = network.getInputsInfo();
    for (const auto & inputInfoItem : input_data_map) {
        if (inputInfoItem.second->getTensorDesc().getDims().size() == 4) {  // first input contains images
            imageInputName = inputInfoItem.first;
            inputInfoItem.second->setPrecision(InferenceEngine::Precision::U8);
            inputInfoItem.second->getInputData()->setLayout(InferenceEngine::Layout::NCHW);
            const InferenceEngine::TensorDesc& inputDesc = inputInfoItem.second->getTensorDesc();
            inputLayerSize = cv::Size(getTensorWidth(inputDesc), getTensorHeight(inputDesc));
        } else if (inputInfoItem.second->getTensorDesc().getDims().size() == 2) {  // second input contains image info
            imageInfoInputName = inputInfoItem.first;
            inputInfoItem.second->setPrecision(InferenceEngine::Precision::FP32);
        } else {
            throw std::logic_error("Unsupported " +
                                   std::to_string(inputInfoItem.second->getTensorDesc().getDims().size()) + "D "
                                   "input layer '" + inputInfoItem.first + "'. "
                                   "Only 2D and 4D input layers are supported");
        }
    }

    // prepare output blobs
    InferenceEngine::OutputsDataMap outputInfo = network.getOutputsInfo();
    if (outputInfo.size() != 1) {
        throw std::logic_error("Expected SSD network with only one output");
    }
    auto outputBlobsIt = outputInfo.begin();
    InferenceEngine::DataPtr& output = outputBlobsIt->second;
    outputName = outputBlobsIt->first;
    const InferenceEngine::SizeVector outputDims = output->getTensorDesc().getDims();
    maxProposalCount = outputDims[2];
    objectSize = outputDims[3];
    if (objectSize != 7) {
        throw std::logic_error("Output should have 7 as a last dimension");
    }
    if (outputDims.size() != 4) {
        throw std::logic_error("Incorrect output dimensions for SSD");
    }
    output->setPrecision(InferenceEngine::Precision::FP32);
    output->setLayout(InferenceEngine::Layout::NCHW);

    // it's enough just to set image info input (if used in the model) only once
    auto setImgInfoBlob = [&](const InferenceEngine::InferRequest::Ptr &inferReq) {
        auto blob = inferReq->GetBlob(imageInfoInputName);
        auto data = blob->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type *>();
        data[0] = static_cast<float>(inputLayerSize.height);  // height
        data[1] = static_cast<float>(inputLayerSize.width);  // width
        data[2] = 1;
    };

    // load network to device
    for (int i = 0; i < numDevices; i++) {
        auto w = std::make_shared<Worker>(i, queueSize_, availableDevices.at(i));
        //std::cout << "estimator " << i << " targetDeviceName : " << w->target_device_name << std::endl;
        w->executableNetwork = ie.LoadNetwork(network, w->target_device_name, {});
        // TODO: move this loop into worker object
        for (int j = 0; j < w->queue_size; j++) {
            w->infwork[j].first = w->executableNetwork.CreateInferRequestPtr();
            if (!imageInfoInputName.empty())
                setImgInfoBlob(w->infwork[j].first);
            set_notify_on_job_completion(&w->infwork[j], w->jobs_mutex, w->worker_id);
        }
        workers.push_back(w);
    }
}


SSDetector::~SSDetector() {
    // TODO: remove callbacks
    // SetCompletionCallback
    for (auto& worker : this->workers) {
        for (int j = 0; j < worker->queue_size; j++) {
            worker->infwork[j].first->SetCompletionCallback([]{});
        }
    }
    // SetLogCallback
}


void SSDetector::getInputWidthHeight(int *width, int *height) {
    *width = inputLayerSize.width;
    *height = inputLayerSize.height;
}


static bool compare_timestamp(std::pair<std::shared_ptr<struct timeval>, std::vector<ssd::SSDObject>> &first,
                                           std::pair<std::shared_ptr<struct timeval>, std::vector<ssd::SSDObject>> &second)
{
    return (timercmp(first.first.get(), second.first.get(), <));
}


void SSDetector::set_notify_on_job_completion(std::pair<InferenceEngine::InferRequest::Ptr, std::shared_ptr<job::Job>> *infwork,
                                                      std::shared_ptr<std::mutex> jobs_mutex,
                                                      int worker_id_) {
    // https://docs.openvinotoolkit.org/latest/classInferenceEngine_1_1IInferRequest.html
    // TODO: SetUserData() and GetUserData()
    //infRequestPtr->SetCompletionCallback(
    //    [](InferenceEngine::IInferRequest::Ptr request, InferenceEngine::StatusCode status)
    //    {
    //    };

    infwork->first->SetCompletionCallback(
        [&, infwork, jobs_mutex, worker_id_] {
                //std::cout << "estimator " << worker_id_ << " callback for job ID " << infwork->second->id << std::endl;

                // Get inference result
                std::unique_lock<std::mutex> mlock(*jobs_mutex);
                if ((infwork->second) && (infwork->second->is_valid())) {
                    std::vector<ssd::SSDObject> objects = getInferenceResult(infwork->first,
                                                                             infwork->second,
                                                                             worker_id_);
                    int job_id = infwork->second->id;
                    std::shared_ptr<struct timeval> ts = infwork->second->timestamp;
                    std::shared_ptr<job::Job> no_job(nullptr);
                    no_job.swap(infwork->second);
                    mlock.unlock();

                    std::unique_lock<std::mutex> resmlock(results_mutex);
                    std::map<int, std::list<std::pair<std::shared_ptr<struct timeval>, std::vector<ssd::SSDObject>>>>::iterator it;
                    it = results.find(job_id);
                    if (it != results.end()) {
                        // found it, append to list
                        it->second.emplace_back(ts, objects);
                        it->second.sort(compare_timestamp);
                    } else {
                        // not found, create a new list
                        std::list<std::pair<std::shared_ptr<struct timeval>, std::vector<ssd::SSDObject>>> result_q;
                        result_q.emplace_back(ts, objects);
                        results.insert(std::make_pair(job_id, result_q));
                    }
                    resmlock.unlock();
                } else {
                    // invalid job ?!
                    //std::cout << "Invalid Job ID" << std::endl;
                    if (InferenceEngine::StatusCode::OK == infwork->first->Wait(InferenceEngine::IInferRequest::WaitMode::STATUS_ONLY))
                        infwork->first->Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
                }
            }
        );
}


int SSDetector::save_image_as_png(const cv::Mat &img, const char *filename)
{
    //cv::Size imageSize = img.size();
    //std::cout << "imageSize.width = " << imageSize.width << std::endl;
    //std::cout << "imageSize.height = " << imageSize.height << std::endl;
    std::vector<int> compression_params;
    compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(0);

    try {
        cv::imwrite(filename, img, compression_params);
    }
    catch (std::runtime_error& ex) {
        errMessage = "failed to convert image to PNG format: ";
        errMessage.append(ex.what());
        return -1;
    }
    return 0;
}


bool SSDetector::queueJob(int id, struct timeval *timestamp,
                                  unsigned char *image, int width, int height)
{
    bool ret = false;

    if (!job::id_is_valid(id))
        return ret;

    std::shared_ptr<Worker> nominated_worker = std::shared_ptr<Worker>(nullptr);
    if (!matchJobIdToWorkerId) {
        // mode 1: push job to next available queue
        // find which worker has the most empty queue
        int max_avail_size = 0;
        for (auto& worker : this->workers) {
            int q_size = worker->queue_available_size();
            if (q_size > max_avail_size) {
                max_avail_size = q_size;
                nominated_worker = worker;
            }
        }
    } else {
        // mode 2: push job to worker with matchin ID
        for (auto& worker : this->workers) {
            if (worker->worker_id == id) {
                if (worker->queue_available_size() > 0)
                    nominated_worker = worker;
                break;
            }
        }
    }

    if (nominated_worker) {
        int scaled_width = 0;
        int scaled_height = 0;
        get_scaled_image_dimensions(width, height, &scaled_width, &scaled_height);

        //// resize model input
        //if ((scaled_width != inputLayerSize.width) || (scaled_height != inputLayerSize.height)) {
        //    inputLayerSize.width = scaled_width;
        //    inputLayerSize.height = scaled_height;
        //    auto input_shapes = network.getInputShapes();
        //    std::string input_name;
        //    InferenceEngine::SizeVector input_shape;
        //    std::tie(input_name, input_shape) = *input_shapes.begin();
        //    input_shape[2] = inputLayerSize.height;
        //    input_shape[3] = inputLayerSize.width;
        //    input_shapes[input_name] = input_shape;
        //    network.reshape(input_shapes);
        //
        //    nominated_worker->executableNetwork = ie.LoadNetwork(network, nominated_worker->target_device_name, {});
        //    for (int j = 0; j < nominated_worker->queue_size; j++) {
        //        nominated_worker->infwork[j].first = nominated_worker->executableNetwork.CreateInferRequestPtr();
        //        set_notify_on_job_completion(&nominated_worker->infwork[j], nominated_worker->jobs_mutex, nominated_worker->worker_id);
        //    }
        //}

        unsigned char *scaled_img = scale_yuv2bgr(image, width, height, scaled_width, scaled_height);
        if (scaled_img) {
            try {
                std::unique_lock<std::mutex> mlock(*nominated_worker->jobs_mutex);
                for (int i = 0; i < nominated_worker->queue_size; i++) {
                    if ((!nominated_worker->infwork[i].second) || (!nominated_worker->infwork[i].second->is_valid())) {
                        std::shared_ptr<struct timeval> ts = std::make_shared<struct timeval>();
                        ts->tv_sec = timestamp->tv_sec;
                        ts->tv_usec = timestamp->tv_usec;
                        nominated_worker->infwork[i].second = std::make_shared<job::Job>(id, ts, width, height, scaled_width, scaled_height, scaled_img);
                        mlock.unlock();
                        estimateAsync(nominated_worker->worker_id,
                                      nominated_worker->infwork[i].first,
                                      nominated_worker->infwork[i].second);
                        ret = true;
                        break;
                    }
                }
            }
            catch (const std::exception &ex) {
                errMessage = "failed to queue inference: ";
                errMessage.append(ex.what());
            }
            av_freep(&scaled_img);
        }
        return ret;
    }

    return ret;
}


// return id negative means no inference result
void SSDetector::estimateAsync(int worker_id_, InferenceEngine::InferRequest::Ptr request, std::shared_ptr<job::Job> the_job) {
    //save_image_as_png(the_job->scaledImage, "scaled_input.png");
    cv::Mat paddedImage = padImage(the_job->scaledImage);
    //save_image_as_png(paddedImage, "padded_input.png");
    InferenceEngine::Blob::Ptr input = request->GetBlob(imageInputName);
    auto buffer = input->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::U8>::value_type *>();
    imageToBuffer(paddedImage, buffer);

    //std::cout << "estimator " << worker_id_ << " : Start async inference for job ID " << the_job->id << std::endl;
    request->StartAsync();
}


void SSDetector::pollAsyncInferenceResults(void)
{
    for (auto& worker : this->workers) {
        for (int i = 0; i < worker->queue_size; i++) {
            std::unique_lock<std::mutex> mlock(*worker->jobs_mutex);
            if ((worker->infwork[i].second) && (worker->infwork[i].second->is_valid())) {

                InferenceEngine::StatusCode state = worker->infwork[i].first->Wait(InferenceEngine::IInferRequest::WaitMode::STATUS_ONLY);
                if (InferenceEngine::StatusCode::OK == state) {
                    std::vector<ssd::SSDObject> objects = getWaitInferenceResult(worker->infwork[i].first,
                                                                                 worker->infwork[i].second,
                                                                                 worker->worker_id);
                    int job_id = worker->infwork[i].second->id;
                    std::shared_ptr<struct timeval> ts = worker->infwork[i].second->timestamp;
                    std::shared_ptr<job::Job> no_job(nullptr);
                    no_job.swap(worker->infwork[i].second);
                    mlock.unlock();

                    std::unique_lock<std::mutex> resmlock(results_mutex);
                    std::map<int, std::list<std::pair<std::shared_ptr<struct timeval>, std::vector<ssd::SSDObject>>>>::iterator it;
                    it = results.find(job_id);
                    if (it != results.end()) {
                        // found it, append to queue
                        it->second.emplace_back(ts, objects);
                        it->second.sort(compare_timestamp);
                    } else {
                        // not found, create a new queue
                        std::list<std::pair<std::shared_ptr<struct timeval>, std::vector<ssd::SSDObject>>> result_q;
                        result_q.emplace_back(ts, objects);
                        results.insert(std::make_pair(job_id, result_q));
                    }
                    resmlock.unlock();
                }
            }
        }
    }
}


bool SSDetector::resultIsReady(int id)
{
    //pollAsyncInferenceResults();
    std::unique_lock<std::mutex> mlock(results_mutex);
    std::map<int, std::list<std::pair<std::shared_ptr<struct timeval>, std::vector<ssd::SSDObject>>>>::iterator it;
    it = results.find(id);
    return (it != results.end());
}


std::pair<std::shared_ptr<struct timeval>, std::vector<ssd::SSDObject>> SSDetector::getResult(int id)
{
    std::pair<std::shared_ptr<struct timeval>, std::vector<ssd::SSDObject>> rp;
    std::unique_lock<std::mutex> mlock(results_mutex);
    std::map<int, std::list<std::pair<std::shared_ptr<struct timeval>, std::vector<ssd::SSDObject>>>>::iterator it;
    it = results.find(id);
    if (it != results.end()) {
        rp = it->second.front();
        it->second.pop_front();
        if (it->second.empty())
            results.erase(id);
    }
    return rp;
}


std::vector<ssd::SSDObject> SSDetector::getObjects(InferenceEngine::InferRequest::Ptr request,
                                                   const cv::Size& orgImageSize,
                                                   const cv::Size& scaledImageSize) const {
    InferenceEngine::Blob::Ptr outputBlob = request->GetBlob(outputName);
    std::vector<ssd::SSDObject> objects;
    const float *detections = outputBlob->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();
    std::set<int>::iterator it;

    // a. First fp16 value holds the number of valid detections = num_valid.
    // b. The next 6 values are unused.
    // c. The next (7 * num_valid) values contain the valid detections data
    //      Each group of 7 values will describe an object/box These 7 values in order.
    //      The values are:
    //        0: image_id (always 0)
    //        1: class_id (this is an index into labels)
    //        2: score (this is the probability for the class)
    //        3: box left location within image as number between 0.0 and 1.0
    //        4: box top location within image as number between 0.0 and 1.0
    //        5: box right location within image as number between 0.0 and 1.0
    //        6: box bottom location within image as number between 0.0 and 1.0

    for (int i = 0; i < maxProposalCount; i++) {
        float image_id = detections[i * objectSize + 0];
        if (image_id < 0) {
            break;
        }
        int label = static_cast<int>(detections[i * objectSize + 1]);
        it = idFilter.find(label);
        if (it == idFilter.end())
            continue;
        float confidence = detections[i * objectSize + 2];
        float xmin = detections[i * objectSize + 3];
        float ymin = detections[i * objectSize + 4];
        float xmax = detections[i * objectSize + 5];
        float ymax = detections[i * objectSize + 6];

        objects.emplace_back(label, confidence, xmin, xmax, ymin, ymax);
    }

    correctCoordinates(objects, orgImageSize, scaledImageSize);

    return objects;
}


std::vector<ssd::SSDObject> SSDetector::getWaitInferenceResult(InferenceEngine::InferRequest::Ptr request,
                                                               std::shared_ptr<job::Job> the_job,
                                                               int worker_id_) {
    std::vector<ssd::SSDObject> objects;
    InferenceEngine::StatusCode state = request->Wait(InferenceEngine::IInferRequest::WaitMode::STATUS_ONLY);
    //std::cout << "estimator " << worker_id_ << " status is " << state << std::endl;
    if (InferenceEngine::StatusCode::OK == state) {
        // process result
        //std::cout << "estimator " << worker_id_ << " : Inference job completed, calling wait" << std::endl;
        if (InferenceEngine::StatusCode::OK == request->Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY)) {

            //std::cout << "estimator " << worker_id_ << " : Getting results for job ID " << the_job->id << std::endl;

            cv::Size scaledImageSize = the_job->scaledImage.size();
            objects = getObjects(request,
                                 the_job->fullImageSize,
                                 scaledImageSize);
        }
    }
    return objects;
}


std::vector<ssd::SSDObject> SSDetector::getInferenceResult(InferenceEngine::InferRequest::Ptr request,
                                                                                     std::shared_ptr<job::Job> the_job,
                                                                                     int worker_id_) {
    //std::cout << "estimator " << worker_id_ << " : Getting results for job ID " << the_job->id << std::endl;

    cv::Size scaledImageSize = the_job->scaledImage.size();
    std::vector<ssd::SSDObject> objects = getObjects(request,
                                                     the_job->fullImageSize,
                                                     scaledImageSize);
    return objects;
}


void SSDetector::imageToBuffer(const cv::Mat& scaledImage, uint8_t* buffer) const {
    std::vector<cv::Mat> planes(3);
    for (size_t pId = 0; pId < planes.size(); pId++) {
        planes[pId] = cv::Mat(inputLayerSize, CV_8UC1,
                              buffer + pId * inputLayerSize.area());
    }
    cv::split(scaledImage, planes);
}


cv::Mat SSDetector::padImage(const cv::Mat& scaledImage) const {
    cv::Mat paddedImage;
    cv::Size scaledImageSize = scaledImage.size();
    int w_diff = inputLayerSize.width - scaledImageSize.width;
    int h_diff = inputLayerSize.height - scaledImageSize.height;
    int left = w_diff >> 1;
    int right = w_diff - left;
    int top = h_diff >> 1;
    int bottom = h_diff - top;
    cv::copyMakeBorder(scaledImage, paddedImage, top, bottom, left, right,
                       cv::BORDER_CONSTANT, meanPixel);
    return paddedImage;
}


float SSDetector::clamp(float v, float lo, float hi) const {
    if (v < lo)
        return lo;
    if (v > hi)
        return hi;
    return v;
}


void SSDetector::correctCoordinates(std::vector<ssd::SSDObject>& objects,
                                    const cv::Size& imageSize,
                                    const cv::Size& scaledImageSize) const {
    int w_diff = inputLayerSize.width - scaledImageSize.width;
    int h_diff = inputLayerSize.height - scaledImageSize.height;
    int left = w_diff >> 1;
    int right = w_diff - left;
    int top = h_diff >> 1;
    int bottom = h_diff - top;

    float scaleX = imageSize.width /
            static_cast<float>(inputLayerSize.width - left - right);
    float scaleY = imageSize.height /
            static_cast<float>(inputLayerSize.height - top - bottom);

    for (auto& obj : objects) {
        obj.xmin = clamp((obj.xmin * inputLayerSize.width  - left) * scaleX, 0, imageSize.width-1);
        obj.xmax = clamp((obj.xmax * inputLayerSize.width  - left) * scaleX, 0, imageSize.width-1);
        obj.ymin = clamp((obj.ymin * inputLayerSize.height - top ) * scaleY, 0, imageSize.height-1);
        obj.ymax = clamp((obj.ymax * inputLayerSize.height - top ) * scaleY, 0, imageSize.height-1);
    }
}



Worker::Worker(int worker_id_, int queueSize_, std::string &targetDeviceName)
{
    worker_id = worker_id_;
    queue_size = queueSize_;
    target_device_name = targetDeviceName;
    jobs_mutex = std::make_shared<std::mutex>();
    infwork = std::make_unique<std::pair<InferenceEngine::InferRequest::Ptr, std::shared_ptr<job::Job>>[]>(queue_size);
}


int Worker::queue_available_size(void)
{
    int cnt = 0;
    std::unique_lock<std::mutex> mlock(*jobs_mutex);
    for (int i = 0; i < queue_size; i++) {
        if ((!infwork[i].second) || (!infwork[i].second->is_valid()))
            cnt++;
    }
    return cnt;
}


}  // namespace ssd
