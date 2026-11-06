package detection

import (
	"errors"
	"image"

	"github.com/oxzjh/cv"
)

type IDetector interface {
	Detect(*cv.Mat) ([]*cv.Box, error)
	Close()
}

type Detection struct {
	detector IDetector
}

func (d *Detection) Detect(m *cv.Mat) ([]*Object, error) {
	boxes, err := d.detector.Detect(m)
	if err != nil {
		return nil, err
	}
	objects := make([]*Object, len(boxes))
	for i, v := range boxes {
		obj := &Object{
			Rect: image.Rect(
				int(v.X0+0.5),
				int(v.Y0+0.5),
				int(v.X1+0.5),
				int(v.Y1+0.5),
			),
			Score: v.Score,
		}
		switch t := v.Data.(type) {
		case float32:
			obj.Class = Class(t)
		case int:
			obj.Class = Class(t)
		}
		objects[i] = obj
	}
	return objects, nil
}

func (d *Detection) DetectFile(file string) ([]*Object, error) {
	m := cv.IMRead(file, cv.FLAG_COLOR)
	defer m.Free()
	if m.Cols() < 4 || m.Rows() < 4 {
		return nil, errors.New("read file failed")
	}
	return d.Detect(m)
}

func (d *Detection) Close() {
	d.detector.Close()
}

func New(detector IDetector) *Detection {
	return &Detection{detector}
}
