import Images 
using  GraphNeuralNetworks

function process_image(img, dims)
    x = Images.RGB(img)
    x = Images.imresize(x, dims[1], dims[2])
    x = Images.channelview(x)
    x = reshape(x, (3, dims[1]*dims[2]))
    return x
end

function rag(dims::Tuple{Int, Int})
    pixel_index(x, y, width) = (y-1)*width + x

    neighbor_offsets = [
        (-1, 0), (1, 0),  # Left and right
        (0, -1), (0, 1),  # Up and down
        (-1, -1), (1, -1),  # Diagonal up-left and up-right
        (-1, 1), (1, 1)   # Diagonal down-left and down-right
    ]
    
    N = dims[1]*dims[2]
    E = 8*N - 6*(dims[1] + dims[2]) + 4
    src, dst = Vector{Int64}(undef, E), Vector{Int64}(undef, E)

    edge_index = 1
    @inbounds for y in 1:dims[1]
        for x in 1:dims[2]
            current_pixel = pixel_index(x, y, dims[2])
            for (dx, dy) in neighbor_offsets
                nx, ny = x + dx, y + dy
                if 1 <= nx <= dims[2] && 1 <= ny <= dims[1]
                    neighbor_pixel = pixel_index(nx, ny, dims[2])
                    src[edge_index] = current_pixel
                    dst[edge_index] = neighbor_pixel
                    edge_index += 1
                end
            end
        end
    end
         
    return GNNGraph(src, dst, num_nodes=N)
end


function compute_edge_weights(g::GNNGraph)
    src, dst = edge_index(g)
    x_dim = size(g.x, 1)

    w = @views sum(abs.(g.x[:, src] .- g.x[:, dst]), dims=1) ./ x_dim
    return w
end

function compute_edge_weights!(g::GNNGraph)
    src, dst = edge_index(g)
    x_dim = size(g.x, 1)

    g.edata.e = @views sum(abs.(g.x[:, src] .- g.x[:, dst]), dims=1) ./ x_dim
end
