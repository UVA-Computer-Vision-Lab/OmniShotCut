import argparse



def get_args_parser(is_testing = False):      # Argument Control

    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default = 1e-4, type=float)
    parser.add_argument('--lr_backbone', default = 1e-5, type=float)
    parser.add_argument('--total_batch_size', default = 64, type=int)                # Batch Size
    parser.add_argument('--weight_decay', default = 1e-4, type=float)
    parser.add_argument('--epochs', default = 100, type=int)                    
    parser.add_argument('--lr_drop', default = 50, type=int)
    parser.add_argument('--clip_max_norm', default = 0.1, type=float,
                        help='gradient clipping max norm')
    

    # Model parameters
    parser.add_argument('--num_intra_relation_classes', default = 10, type = int,
                            help="Number of classes in Transformer for the Intra Clip Relation classes")        
    parser.add_argument('--num_inter_relation_classes', default = 7, type = int,
                            help="Number of classes in Transformer for the Inter Clip Relation classes") 

    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet18', type=str,             
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default = 3, type=int,
                        help="Number of encoding layers in the transformer")                        
    parser.add_argument('--dec_layers', default = 6, type=int,  
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default = 2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default = 384, type=int,
                        help="Size of the embeddings (dimension of the transformer)")               
    parser.add_argument('--dropout', default = 0.0, type=float,         
                        help="Dropout applied in the transformer")                                  
    parser.add_argument('--nheads', default = 8, type=int,                                          
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default = 24, type=int,
                        help="Number of query slots")                                               
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")


    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Loss coefficients
    parser.add_argument('--range_loss_coef', default = 5, type=float)          
    parser.add_argument('--intra_clip_label_loss_coef', default = 1, type=float)       
    parser.add_argument('--inter_clip_label_loss_coef', default = 1, type=float)        


    # Dataset parameters
    parser.add_argument('--dataset_file', default='shot_boundary_detection')            # Dataloader name
    parser.add_argument('--train_data_info_path', required = True if not is_testing else False, type=str)                    # Required: dataset info path (with video path and the transition range)
    parser.add_argument('--process_height', default = 96, type=int)                     # Default Start     128 x 96
    parser.add_argument('--process_width', default = 128, type=int)
    parser.add_argument('--max_process_window_length', default = 100, type=int)         # How many video frames to process, unify the size
    parser.add_argument('--has_overlength_prob', default = 0.1, type=float)             
    parser.add_argument('--min_video_in_padding', default = 30, type=int)                
    parser.add_argument('--val_data_info_path', required = True if not is_testing else False, type=str) 
    parser.add_argument('--max_val_num', default = 200, type=int)
    

    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--output_dir', default='results_training',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default = 2887, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')            # use "latest" / exact path
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default = 8, type=int)                           # Recommend > 6 for workers of the dataset


    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')


    # TensorBoard Setting
    parser.add_argument('--tensorboard_name', default = "exp_trial", type = str,
                        help = 'number of distributed processes')

    return parser