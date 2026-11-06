package detection

import (
	"github.com/oxzjh/cv"
	"github.com/oxzjh/ort"
)

type MobileNet struct {
	engine  *ort.Ort
	maxSize int
	score   float32
	iou     float32
}

func (mn *MobileNet) Detect(m *cv.Mat) (boxes []*cv.Box, err error) {
	originWidth := m.Cols()
	originHeight := m.Rows()
	width := originWidth
	height := originHeight
	if originWidth > originHeight {
		if originWidth > mn.maxSize {
			width = mn.maxSize
			height = mn.maxSize * originHeight / originWidth
		}
	} else if originHeight > mn.maxSize {
		width = mn.maxSize * originWidth / originHeight
		height = mn.maxSize
	}
	inputBlob := m.CvtColor(cv.COLOR_BGR2RGB)
	defer inputBlob.Free()
	var result *ort.Result
	if result, err = mn.engine.InferWithShape(inputBlob.Data(), inputBlob.Total()*inputBlob.ElemSize(), []int64{1, int64(height), int64(width), 3}, ort.TypeUint8); err != nil {
		return
	}
	defer result.Free()
	var numArr []float32
	if numArr, err = result.GetFloat32(3, 1, false); err != nil {
		return
	}
	num := int(numArr[0])
	var (
		bboxes  []float32
		classes []float32
		scores  []float32
	)
	if bboxes, err = result.GetFloat32(0, num<<2, false); err != nil {
		return
	}
	if classes, err = result.GetFloat32(1, num, false); err != nil {
		return
	}
	if scores, err = result.GetFloat32(2, num, false); err != nil {
		return
	}
	for i := 0; i < num; i++ {
		if score := scores[i]; score > mn.score {
			boxOffset := i << 2
			boxes = append(boxes, &cv.Box{
				ID:    i,
				X0:    bboxes[boxOffset+1] * float32(originWidth),
				Y0:    bboxes[boxOffset] * float32(originHeight),
				X1:    bboxes[boxOffset+3] * float32(originWidth),
				Y1:    bboxes[boxOffset+2] * float32(originHeight),
				Score: score,
				Data:  classes[i],
			})
		} else {
			break
		}
	}
	boxes = cv.NMS(boxes, mn.iou)
	return
}

func (mn *MobileNet) Close() {
	if mn.engine != nil {
		mn.engine.Close()
		mn.engine = nil
	}
}

func NewMobileNet(model string, maxSize /*= 640*/ int, score /*= 0.45*/, iou /*= 0.25*/ float32, threads int) (*MobileNet, error) {
	engine, err := ort.New(model, "detection_mobile_net", threads)
	if err != nil {
		return nil, err
	}
	return &MobileNet{engine, maxSize, score, iou}, nil
}

func NewDefaultMobileNet(model string, threads int) (*MobileNet, error) {
	return NewMobileNet(model, 640, 0.45, 0.25, threads)
}
