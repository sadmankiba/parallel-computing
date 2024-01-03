#ifndef CONV_H
#define CONV_H


/* NOTE: Assumes that input and current layer is of 1 channel */
class Conv: public BaseLayer {
private:
    V3D input_last;
    V3D a_last;
    V3D grad_z_cur; /* dim 1 - channel, dim 2 - img x, dim 3 - img y */

    void init_weights() override;
    void init_bias();

    V2D rot_180(const V2D& input);

    V2D pad(const V2D& input, int pad_rows, int pad_cols);

public:
    int fs; /* size of the filter, assuming square */
    int chn_prev; /* number of channels in previous layer */
    int chn_cur; /* number of channels in current layer */
    V3D weights; /* Assuming input is of 1-channel */
    V1D bias; /* per cur channel */
    long long conv_time;

    Conv(int _chn_prev, int _chn_cur, int _fs, bool _use_cpu);

    V2D conv(V2D input, V2D filter);
    
    V3D forward(V3D input);

    V3D calc_grad_z(V3D w_nxt, V3D grad_z_nxt);

    void update_weights();

    void set_in_size(std::vector<int> _in_size); 

    void set_grad_z(V3D grad_z);

    void print_weights();
};



#endif