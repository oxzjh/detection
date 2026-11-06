package detection

import (
	"github.com/oxzjh/cv"
	"github.com/oxzjh/ort"
)

type YoloV6 struct {
	engine *ort.Ort
	width  int
	height int
	rows   int
	cols   int
	score  float32
	iou    float32
}

func (yo *YoloV6) Detect(m *cv.Mat) (boxes []*cv.Box, err error) {
	resized, border := m.ResizeWithBorder(yo.width, yo.height, cv.INTER_CUBIC, cv.BORDER_CONSTANT, cv.NewScalar(114, 114, 114, 0))
	inputBlob := resized.BlobFromImage(1/255.0, cv.Size{}, cv.Scalar{}, true, false)
	resized.Free()
	defer inputBlob.Free()
	var result *ort.Result
	if result, err = yo.engine.Infer(inputBlob.Data(), inputBlob.Total()*inputBlob.ElemSize(), ort.TypeFloat); err != nil {
		return
	}
	defer result.Free()
	var data []float32
	if data, err = result.GetFloat32(0, yo.rows*yo.cols, false); err != nil {
		return
	}
	for i := 0; i < yo.rows; i++ {
		offset := i * yo.cols
		if score := data[offset+4]; score > yo.score {
			classScore := data[offset+5]
			classIndex := 0
			for j := 6; j < yo.cols; j++ {
				if s := data[offset+j]; s > classScore {
					classScore = s
					classIndex = j - 5
				}
			}
			score *= classScore
			if score > yo.score {
				cx := data[offset]
				cy := data[offset+1]
				hw := data[offset+2] * 0.5
				hh := data[offset+3] * 0.5
				boxes = append(boxes, &cv.Box{
					ID:    offset,
					X0:    cv.Maxf(0, cx-hw),
					Y0:    cv.Maxf(0, cy-hh),
					X1:    cv.Minf(cx+hw, float32(yo.width)),
					Y1:    cv.Minf(cy+hh, float32(yo.height)),
					Score: score,
					Data:  classIndex,
				})
			}
		}
	}
	boxes = cv.NMS(boxes, yo.iou)
	if len(boxes) > 0 {
		ratio := 1 / border.Ratio
		for _, box := range boxes {
			box.X0 = (box.X0 - float32(border.Left)) * ratio
			box.Y0 = (box.Y0 - float32(border.Top)) * ratio
			box.X1 = (box.X1 - float32(border.Left)) * ratio
			box.Y1 = (box.Y1 - float32(border.Top)) * ratio
		}
	}
	return
}

func (yo *YoloV6) Close() {
	if yo.engine != nil {
		yo.engine.Close()
		yo.engine = nil
	}
}

func NewYoloV6(model string, score /*= 0.45*/, iou /*= 0.25*/ float32, threads int) (*YoloV6, error) {
	engine, err := ort.New(model, "detection_yolov6", threads)
	if err != nil {
		return nil, err
	}
	return &YoloV6{engine, int(engine.InputDims[0][3]), int(engine.InputDims[0][2]), int(engine.OutputDims[0][1]), int(engine.OutputDims[0][2]), score, iou}, nil
}

func NewDefaultYolov6(model string, threads int) (*YoloV6, error) {
	return NewYoloV6(model, 0.45, 0.25, threads)
}
